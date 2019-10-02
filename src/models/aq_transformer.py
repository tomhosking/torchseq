import torch.nn as nn
import torch

from utils.bpe_factory import BPE

from models.positional_embeddings import SinusoidalPositionalEmbedding

class TransformerAqModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.embeddings = nn.Embedding.from_pretrained(torch.Tensor(BPE.instance().emb.vectors), freeze=True).cpu() # TODO: this should come from a config
        self.bio_embeddings = nn.Embedding(3, config.embedding_dim).cpu()
        
        # self.embeddings.weight = nn.Parameter(, requires_grad=False)

        self.encoder_decoder = nn.Transformer(d_model=config.embedding_dim,
                                                nhead=config.encdec.num_heads,
                                                num_encoder_layers=config.encdec.num_encoder_layers,
                                                num_decoder_layers=config.encdec.num_decoder_layers,
                                                dim_feedforward=config.encdec.dim_feedforward)

        self.output_projection = nn.Linear(config.embedding_dim, config.vocab_size+1)
        # self.input_projection = nn.Linear(config.embedding_dim+config.bio_embedding_dim, config.embedding_dim)

        # Init output projection layer with embedding matrix
        self.output_projection.weight.data = self.embeddings.weight.data

        self.positional_embeddings = SinusoidalPositionalEmbedding(embedding_dim=config.embedding_dim, padding_idx=-1, init_size=400)
        


    def forward(self, batch, output):

        max_ctxt_len = torch.max(batch['c_len'])
        max_q_len = torch.max(batch['q_len'])
        curr_batch_size = batch['c'].size()[0]
        

        output_max_len = output.size()[-1]
        tgt_mask = torch.FloatTensor(output_max_len, output_max_len).fill_(float('-inf')).to(self.device)
        tgt_mask = torch.triu(tgt_mask, diagonal=1)

        # ie how many indices are non-pad
        output_len = torch.sum(torch.ne(output, BPE.pad_id), dim=-1)

        context_mask = (torch.arange(max_ctxt_len)[None, :].cpu() > batch['c_len'][:, None].cpu()).to(self.device)
        output_pad_mask = (torch.arange(output_max_len)[None, :].cpu() > output_len[:, None].cpu()).to(self.device)[:, :output_max_len]


        # print(output_pad_mask.shape)
        # print(batch['a_pos'][0])

        ctxt_toks_embedded = self.embeddings(batch['c']).to(self.device)
        ctxt_ans_embedded = self.bio_embeddings(batch['a_pos']).to(self.device)
        # print(ctxt_toks_embedded.shape, ctxt_ans_embedded.shape, ctxt_embedded.shape)
        # print(self.positional_embeddings.weights.shape)
        output_embedded = self.embeddings(output).to(self.device)

        # print(ctxt_embedded.shape)

        ctxt_pos_embeddings = self.positional_embeddings(batch['c'])
        output_pos_embeddings = self.positional_embeddings(output)
        

        # print(ctxt_pos_embeddings.shape)

        ctxt_embedded = ctxt_toks_embedded + ctxt_pos_embeddings + ctxt_ans_embedded
        output_embedded += output_pos_embeddings

        # For some reason the Transformer implementation expects seq x batch x feat - this is weird, so permute the input and the output
        output = self.encoder_decoder(src=ctxt_embedded.permute(1,0,2),
                                     tgt=output_embedded.permute(1,0,2),
                                     tgt_mask=tgt_mask,
                                     src_key_padding_mask=context_mask,
                                     tgt_key_padding_mask=output_pad_mask
                                     ).permute(1,0,2)

        
        logits = self.output_projection(output)
        
        
        return logits