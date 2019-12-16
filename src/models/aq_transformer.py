import torch.nn as nn
import torch

import math

from utils.bpe_factory import BPE

from models.positional_embeddings import PositionalEncoding

from models.pooling import MultiHeadedPooling

from transformers import BertModel


def generate_square_subsequent_mask(sz):
    r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

class TransformerAqModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Embedding layers
        # self.embeddings = nn.Embedding.from_pretrained(torch.Tensor(BPE.embeddings), freeze=config.freeze_embeddings).cpu() # TODO: this should come from a config
        self.embeddings = nn.Embedding(config.prepro.vocab_size, config.raw_embedding_dim).cpu() # TODO: this should come from a config
        self.embeddings.weight.data = BPE.instance().embeddings
        self.embeddings.weight.requires_grad = not config.freeze_embeddings

        self.embedding_projection = nn.utils.weight_norm(nn.Linear(config.raw_embedding_dim, config.embedding_dim, bias=False))
        self.bert_embedding_projection = nn.utils.weight_norm(nn.Linear(config.embedding_dim*1+config.bio_embedding_dim, config.embedding_dim, bias=False))
        # self.dec_embedding_projection = nn.Linear(config.raw_embedding_dim, config.embedding_dim, bias=False)

        
        self.bio_embeddings = nn.Embedding.from_pretrained(torch.eye(config.bio_embedding_dim), freeze=True).cpu() if config.onehot_bio else nn.Embedding(3, config.bio_embedding_dim).cpu()

        # Encoder/decoders
        if config.encdec.bert_encoder:
            with torch.no_grad():
                self.bert_encoder = BertModel.from_pretrained(config.encdec.bert_model)
            # self.encoder = DistilBertModel.from_pretrained('distilbert-base-uncased')

            self.freeze_bert = True

            for param in self.bert_encoder.parameters():
                param.requires_grad = False
        
        encoder_layer = nn.TransformerEncoderLayer(config.embedding_dim + (0 if config.encdec.bert_encoder else config.bio_embedding_dim), nhead=config.encdec.num_heads, dim_feedforward=config.encdec.dim_feedforward, dropout=config.dropout, activation=config.encdec.activation)
        encoder_norm = nn.LayerNorm(config.embedding_dim+(0 if config.encdec.bert_encoder else config.bio_embedding_dim))
        self.encoder = nn.TransformerEncoder(encoder_layer, config.encdec.num_encoder_layers, encoder_norm)

        decoder_layer = nn.TransformerDecoderLayer(config.embedding_dim, nhead=config.encdec.num_heads, dim_feedforward=config.encdec.dim_feedforward, dropout=config.dropout, activation=config.encdec.activation)
        decoder_norm = nn.LayerNorm(config.embedding_dim)
        self.decoder = nn.TransformerDecoder(decoder_layer, config.encdec.num_decoder_layers, decoder_norm)

        # Encoder combination
        num_encoder_outputs = sum([1 if v else 0 for k,v in config.encoder_outputs.data.items() if k != 'c_ans_labels'])
        memory_dim = (config.embedding_dim+ (0 if config.encdec.bert_encoder else config.bio_embedding_dim))*num_encoder_outputs
        memory_dim += self.config.bio_embedding_dim if self.config.encoder_outputs.c_ans_labels else 0
        self.encoder_projection = nn.utils.weight_norm(nn.Linear(memory_dim, config.embedding_dim, bias=False))

        
        self.output_projection = nn.Linear(config.embedding_dim, config.prepro.vocab_size, bias=False).cpu()

        # Force the various projections to be unit norm so they can't block information flow
        # with torch.no_grad():
        #     self.embedding_projection.weight.div_(torch.norm(self.embedding_projection.weight, dim=1, keepdim=True))
        #     self.bert_embedding_projection.weight.div_(torch.norm(self.bert_embedding_projection.weight, dim=1, keepdim=True))
        #     self.encoder_projection.weight.div_(torch.norm(self.encoder_projection.weight, dim=1, keepdim=True))
        
        # Init output projection layer with embedding matrix
        if config.embedding_dim == config.raw_embedding_dim:
            self.output_projection.weight.data = self.embeddings.weight.data
        self.output_projection.weight.requires_grad = not config.freeze_projection

        # Pooling layers
        self.ans_pooling = MultiHeadedPooling(config.encdec.num_heads, config.embedding_dim+config.bio_embedding_dim, dropout=config.dropout, model_dim_out=config.embedding_dim, use_final_linear=False)
        self.ctxt_pooling = MultiHeadedPooling(config.encdec.num_heads, config.embedding_dim+config.bio_embedding_dim, dropout=config.dropout, model_dim_out=config.embedding_dim, use_final_linear=False, use_bilinear=True)

        # Position encoding
        self.positional_embeddings_enc = PositionalEncoding(config.embedding_dim+ (0 if config.encdec.bert_encoder else config.bio_embedding_dim))
        self.positional_embeddings_dec = PositionalEncoding(config.embedding_dim)

        
        


    def forward(self, batch, output, memory=None):

        # Re-normalise the projections...
        with torch.no_grad():
            self.embedding_projection.weight_g.div_(self.embedding_projection.weight_g)
            self.bert_embedding_projection.weight_g.div_(self.bert_embedding_projection.weight_g)
            self.encoder_projection.weight_g.div_(self.encoder_projection.weight_g)

        # print(BPE.decode(batch['a'][0][:batch['a_len'][0]]), [BPE.instance().decode([x.item()])  for i,x in enumerate(batch['c'][0]) if batch['a_pos'][0][i].item() > 0], BPE.decode(batch['q'][0][:batch['q_len'][0]]))
        # print([BPE.instance().decode([x.item()])+'/'+str(batch['a_pos'][0][i].item())  for i,x in enumerate(batch['c'][0])])
        # exit()
        # Get some sizes
        max_ctxt_len = torch.max(batch['c_len'])
        max_q_len = torch.max(batch['q_len'])
        curr_batch_size = batch['c'].size()[0]
        output_max_len = output.size()[-1]

        # First pass? Construct the encoding
        if memory is None:
            src_mask = torch.FloatTensor(max_ctxt_len, max_ctxt_len).fill_(float('-inf') if self.config.directional_masks else 0.0).to(self.device)
            src_mask = torch.triu(src_mask, diagonal=1)
            # src_mask = src_mask.where(batch['a_pos'] > 0, torch.zeros_like(src_mask).unsqueeze(-1))

            context_mask = (torch.arange(max_ctxt_len)[None, :].cpu() >= batch['c_len'][:, None].cpu()).to(self.device)


            ctxt_toks_embedded = self.embeddings(batch['c']).to(self.device)
            ctxt_ans_embedded = self.bio_embeddings(batch['a_pos']).to(self.device)

            # Build the context
            if self.config.encdec.bert_encoder:
                ctxt_embedded = ctxt_toks_embedded * math.sqrt(self.config.embedding_dim)
            else:
                if self.config.raw_embedding_dim != self.config.embedding_dim:
                    ctxt_toks_embedded = self.embedding_projection(ctxt_toks_embedded)

                ctxt_embedded = torch.cat([ctxt_toks_embedded, ctxt_ans_embedded], dim=-1) * math.sqrt(self.config.embedding_dim)

                
            
            ctxt_embedded = self.positional_embeddings_enc(ctxt_embedded.permute(1,0,2))
            
            # Fwd pass through encoder
            if self.config.encdec.bert_encoder:
                # print(batch['c'].shape)
                # print(batch['a_pos'].shape)
                # print(context_mask.shape)
                # exit()

                if self.freeze_bert:
                    self.bert_encoder.eval()
                    with torch.no_grad():
                        bert_encoding = self.bert_encoder(input_ids=batch['c'].to(self.device), attention_mask=context_mask)[0] #, token_type_ids=batch['a_pos'].to(self.device)
                else:
                    bert_encoding = self.bert_encoder(input_ids=batch['c'].to(self.device), attention_mask=context_mask)[0] #, token_type_ids=batch['a_pos'].to(self.device)

                if self.config.encdec.num_encoder_layers > 0:
                    bert_encoding_augmented = torch.cat([bert_encoding, ctxt_ans_embedded], dim=-1) # ctxt_embedded.permute(1,0,2)
                    bert_encoding_augmented = self.bert_embedding_projection(bert_encoding_augmented)
                    encoding = self.encoder(bert_encoding_augmented.permute(1,0,2), mask=src_mask, src_key_padding_mask=context_mask).permute(1,0,2).contiguous()
                else:
                    encoding = bert_encoding
                # print(encoding.shape)
            else:
                encoding = self.encoder(ctxt_embedded, mask=src_mask, src_key_padding_mask=context_mask).permute(1,0,2).contiguous()

            # Construct the encoder output by combining a few diff sources
            memory_elements = []
            ans_mask = batch['a_pos'] == 0
            if self.config.encoder_outputs.c_raw:
                memory_elements.append(ctxt_embedded.permute(1,0,2))

            if self.config.encoder_outputs.a_raw:
                memory_elements.append(ctxt_embedded.permute(1,0,2).masked_fill(ans_mask, 0))

            if self.config.encoder_outputs.c_enc:
                memory_elements.append(encoding)

            if self.config.encoder_outputs.c_enc_pool:
                ctxt_pooled = self.ctxt_pooling(key=encoding, value=encoding).unsqueeze(1)
                memory_elements.append(ctxt_pooled.expand(-1, max_ctxt_len, -1))

            if self.config.encoder_outputs.a_enc:
                memory_elements.append(encoding.masked_fill(ans_mask, 0))
            
            if self.config.encoder_outputs.c_ans_labels:
                memory_elements.append(ctxt_ans_embedded)

            if self.config.encoder_outputs.a_enc_pool:
                ans_pooled = self.ans_pooling(encoding, encoding, mask=ans_mask).unsqueeze(1)
                memory_elements.append(ans_pooled.expand(-1, max_ctxt_len, -1))

            

            # This one needs work...
            if self.config.encoder_outputs.c_enc_anspool:
                ans_pooled = self.ans_pooling(encoding, encoding, mask=ans_mask).unsqueeze(1)
                ctxt_anspooled = self.ctxt_pooling(key=ans_pooled, value=encoding).unsqueeze(1)
                memory_elements.append(ctxt_anspooled.expand(-1, max_ctxt_len, -1))
                
            
            memory_full = torch.cat(memory_elements, dim=-1) #, encoding, ctxt_embedded.permute(1,0,2), memory_pooled.expand(-1, max_ctxt_len, -1)
            
            if len(memory_elements) > 1:
                memory = self.encoder_projection(memory_full)
            else:
                memory = memory_full


        # Build some masks
        tgt_mask = torch.FloatTensor(output_max_len, output_max_len).fill_(float('-inf')).to(self.device)
        tgt_mask = torch.triu(tgt_mask, diagonal=1)
        # tgt_mask = generate_square_subsequent_mask(output_max_len).to(self.device)
        # src_mask = generate_square_subsequent_mask(max_ctxt_len).to(self.device)

        # ie how many indices are non-pad
        output_len = torch.sum(torch.ne(output, BPE.pad_id), dim=-1)

    
        output_pad_mask = (torch.arange(output_max_len)[None, :].cpu() >= output_len[:, None].cpu()).to(self.device)[:, :output_max_len]

        # Embed the output so far, then do a decoder fwd pass
        output_embedded = self.embeddings(output).to(self.device) * math.sqrt(self.config.embedding_dim)

        if self.config.raw_embedding_dim != self.config.embedding_dim:
            output_embedded = self.embedding_projection(output_embedded)
        
        # For some reason the Transformer implementation expects seq x batch x feat - this is weird, so permute the input and the output
        output_embedded = self.positional_embeddings_dec(output_embedded.permute(1,0,2))
        
        output = self.decoder(
                                output_embedded,
                                memory.permute(1,0,2),
                                tgt_mask=tgt_mask,
                                tgt_key_padding_mask=output_pad_mask
                            ).permute(1,0,2)

        logits = self.output_projection(output)
        
        return logits, memory