import torch.nn as nn
import torch

import math



from utils.bpe_factory import BPE

# from models.positional_embeddings import SinusoidalPositionalEmbedding
from models.positional_embeddings import PositionalEncoding

from models.pooling import MultiHeadedPooling


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

        self.embeddings = nn.Embedding.from_pretrained(torch.Tensor(BPE.instance().emb.vectors), freeze=config.freeze_embeddings).cpu() # TODO: this should come from a config
        # self.bio_embeddings = nn.Embedding(3, config.embedding_dim).cpu()
        self.bio_embeddings = nn.Embedding.from_pretrained(torch.eye(config.bio_embedding_dim), freeze=True).cpu()
        
        # self.embeddings.weight = nn.Parameter(, requires_grad=False)

        # self.encoder_decoder = nn.Transformer(d_model=config.embedding_dim,
        #                                         nhead=config.encdec.num_heads,
        #                                         num_encoder_layers=config.encdec.num_encoder_layers,
        #                                         num_decoder_layers=config.encdec.num_decoder_layers,
        #                                         dim_feedforward=config.encdec.dim_feedforward,
        #                                         dropout=config.dropout,
        #                                         activation=config.encdec.activation)


        encoder_layer = nn.TransformerEncoderLayer(config.embedding_dim+config.bio_embedding_dim, nhead=config.encdec.num_heads, dim_feedforward=config.encdec.dim_feedforward, dropout=config.dropout, activation=config.encdec.activation)
        encoder_norm = nn.LayerNorm(config.embedding_dim+config.bio_embedding_dim)
        self.encoder = nn.TransformerEncoder(encoder_layer, config.encdec.num_encoder_layers, encoder_norm)
        decoder_layer = nn.TransformerDecoderLayer(config.embedding_dim, nhead=config.encdec.num_heads, dim_feedforward=config.encdec.dim_feedforward, dropout=config.dropout, activation=config.encdec.activation)
        decoder_norm = nn.LayerNorm(config.embedding_dim)
        self.decoder = nn.TransformerDecoder(decoder_layer, config.encdec.num_decoder_layers, decoder_norm)

        self.output_projection = nn.Linear(config.embedding_dim, config.vocab_size+1, bias=False)
        # self.input_projection = nn.Linear(config.embedding_dim+config.bio_embedding_dim, config.embedding_dim)

        # Init output projection layer with embedding matrix
        self.output_projection.weight.data[:, :config.embedding_dim] = self.embeddings.weight.data
        self.output_projection.weight.requires_grad = not config.freeze_projection

        self.pooling = MultiHeadedPooling(config.encdec.num_heads, config.embedding_dim+config.bio_embedding_dim, dropout=config.dropout, model_dim_out=config.embedding_dim)

        # self.positional_embeddings = SinusoidalPositionalEmbedding(embedding_dim=config.embedding_dim, padding_idx=-1, init_size=400)
        self.positional_embeddings_enc = PositionalEncoding(config.embedding_dim+config.bio_embedding_dim)
        self.positional_embeddings_dec = PositionalEncoding(config.embedding_dim)

        self.pool_key = nn.Parameter(torch.randn(config.embedding_dim))
        


    def forward(self, batch, output, memory=None):

        # print(BPE.instance().decode_ids(batch['a'][0][:batch['a_len'][0]]), [BPE.instance().words[x]  for i,x in enumerate(batch['c'][0]) if batch['a_pos'][0][i].item() > 0], BPE.instance().decode_ids(batch['q'][0][:batch['q_len'][0]]))
        

        max_ctxt_len = torch.max(batch['c_len'])
        max_q_len = torch.max(batch['q_len'])
        curr_batch_size = batch['c'].size()[0]


        if memory is None:
            src_mask = torch.FloatTensor(max_ctxt_len, max_ctxt_len).fill_(float('-inf') if self.config.directional_masks else 0.0).to(self.device)
            src_mask = torch.triu(src_mask, diagonal=1)
            

            context_mask = (torch.arange(max_ctxt_len)[None, :].cpu() >= batch['c_len'][:, None].cpu()).to(self.device)


            ctxt_toks_embedded = self.embeddings(batch['c']).to(self.device)
            ctxt_ans_embedded = self.bio_embeddings(batch['a_pos']).to(self.device)

            # ctxt_embedded = ctxt_toks_embedded  + ctxt_ans_embedded

            ctxt_embedded = torch.cat([ctxt_toks_embedded, ctxt_ans_embedded], dim=-1) * math.sqrt(self.config.embedding_dim)

            ctxt_embedded = self.positional_embeddings_enc(ctxt_embedded.permute(1,0,2))

            memory = self.encoder(ctxt_embedded, mask=src_mask, src_key_padding_mask=context_mask).permute(1,0,2)
            

        output_max_len = output.size()[-1]
        tgt_mask = torch.FloatTensor(output_max_len, output_max_len).fill_(float('-inf')).to(self.device)
        tgt_mask = torch.triu(tgt_mask, diagonal=1)
        # tgt_mask = generate_square_subsequent_mask(output_max_len).to(self.device)

        # src_mask = generate_square_subsequent_mask(max_ctxt_len).to(self.device)

        # ie how many indices are non-pad
        output_len = torch.sum(torch.ne(output, BPE.pad_id), dim=-1)

        
        output_pad_mask = (torch.arange(output_max_len)[None, :].cpu() >= output_len[:, None].cpu()).to(self.device)[:, :output_max_len]

        # print(output_pad_mask)

        # print(batch['a_pos'][0])

        # output_ans_embeddings = self.bio_embeddings(torch.zeros_like(output)).to(self.device)
        # print(ctxt_toks_embedded.shape, ctxt_ans_embedded.shape, ctxt_embedded.shape)
        # print(self.positional_embeddings.weights.shape)
        output_embedded = self.embeddings(output).to(self.device) * math.sqrt(self.config.embedding_dim)
        

        # ctxt_pos_embeddings = self.positional_embeddings(batch['c'])
        # output_pos_embeddings = self.positional_embeddings(output)
        

        # print(tgt_mask)

        # ctxt_embedded = torch.cat([ctxt_toks_embedded,ctxt_ans_embedded+ ctxt_pos_embeddings], dim=-1) # + 
        # output_embedded = torch.cat([output_embedded, output_ans_embeddings + output_pos_embeddings], dim=-1)
        
        output_embedded = output_embedded

        

        # print(ctxt_embedded.shape)
        # print(output_embedded.shape)

        # For some reason the Transformer implementation expects seq x batch x feat - this is weird, so permute the input and the output

        memory_pooled = self.pooling(memory, memory).unsqueeze(1)
        
        
        output_embedded = self.positional_embeddings_dec(output_embedded.permute(1,0,2))
        # output = self.encoder_decoder(src=ctxt_embedded,
        #                              tgt=output_embedded,
        #                              tgt_mask=tgt_mask,
        #                              src_mask=src_mask,
        #                              src_key_padding_mask=context_mask,
        #                              tgt_key_padding_mask=output_pad_mask
        #                              ).permute(1,0,2)

        # memory_pooled = memory[batch['c_len']-1, :, :].unsqueeze(0)
        

        output = self.decoder(
                                output_embedded,
                                memory_pooled.permute(1,0,2),
                                tgt_mask=tgt_mask,
                                tgt_key_padding_mask=output_pad_mask
                            ).permute(1,0,2)

        
        logits = self.output_projection(output)
        
        
        return logits, memory