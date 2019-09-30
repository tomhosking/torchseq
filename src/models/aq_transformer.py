import torch.nn as nn
import torch

from utils.bpe_factory import BPE

from models.positional_embeddings import SinusoidalPositionalEmbedding

class TransformerAqModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.embeddings = nn.Embedding.from_pretrained(torch.Tensor(BPE.instance().emb.vectors), freeze=True).cpu() # TODO: this should come from a config
        
        # self.embeddings.weight = nn.Parameter(, requires_grad=False)

        self.encoder_decoder = nn.Transformer(d_model=config.embedding_dim,
                                                nhead=1,
                                                num_encoder_layers=2,
                                                num_decoder_layers=2,
                                                dim_feedforward=256)

        self.output_projection = nn.Linear(config.embedding_dim, config.vocab_size+1)

        # Init output projection layer with embedding matrix
        self.output_projection.weight.data = self.embeddings.weight.data

        self.positional_embeddings = SinusoidalPositionalEmbedding(embedding_dim=config.embedding_dim, padding_idx=-1, init_size=250)


    def forward(self, batch, output):

        # !!!!!!!
        # Set to use question as input for testing
        max_ctxt_len = torch.max(batch['q_len'])
        max_q_len = torch.max(batch['q_len'])
        curr_batch_size = batch['c'].size()[0]
        

        output_max_len = output.size()[-1]
        tgt_mask = torch.FloatTensor(output_max_len, output_max_len).fill_(float('-inf')).to(self.device)
        tgt_mask = torch.triu(tgt_mask, diagonal=1)

        # print(tgt_mask)

        # ie how many indices are non-pad
        output_len = torch.sum(torch.ne(output, BPE.pad_id), dim=-1)
        # print(output_len)

        # print(tgt_mask.shape)

        # !!!!!!!
        # Set to use question as input for testing
        context_mask = (torch.arange(max_ctxt_len)[None, :].cpu() > batch['q_len'][:, None].cpu()).to(self.device)
        output_pad_mask = (torch.arange(output_max_len)[None, :].cpu() > output_len[:, None].cpu()).to(self.device)[:, :output_max_len]


        # print(output_pad_mask.shape)

        # !!!!!!!
        # Set to use question as input for testing
        ctxt_embedded = self.embeddings(batch['q'])
        output_embedded = self.embeddings(output)

        # print(ctxt_embedded.shape)

        ctxt_pos_embeddings = self.positional_embeddings(batch['q'])
        output_pos_embeddings = self.positional_embeddings(output)

        # print(ctxt_pos_embeddings.shape)

        ctxt_embedded += ctxt_pos_embeddings
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