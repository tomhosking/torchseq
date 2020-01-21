import torch
import torch.nn as nn

from utils.bpe_factory import BPE

from pretrained.qa import PreTrainedQA

from utils.metrics import f1

class RerankingReducer(nn.Module):
    def __init__(self, config, device, top1=True):
        super(RerankingReducer, self).__init__()
        self.config = config
        self.device = device

        if config.data.get('reranker', None) is None:
            self.strategy = None
        else:
            self.strategy = config.reranker.strategy

            if self.strategy == 'qa':
                self.qa_model = PreTrainedQA(device=self.device)

    def forward(self, candidates, lengths, batch, tgt_field, scores=None, presorted=True, top1=True):
        curr_batch_size = batch[[k for k in batch.keys()][0]].size()[0]
        
        # Pass-through mode: take the top-1 from a pre-sorted set of candidates (eg beam search)
        if top1 and presorted and self.strategy is None:
            output = candidates[:, 0, :]
            output_lens = lengths[:, 0]

            return output, output_lens, scores

        elif top1 and self.strategy is None:
            # TODO: sort by score and return top-1
            
            return None

        # TODO: strategic reranking goes here

        if self.strategy == 'qa':
            
            # First, stringify
            output_strings = [[BPE.decode(candidates.data[i][j][:lengths[i][j]]) for j in range(len(lengths[i]))]  for i in range(len(lengths))]

            scores = []
            for ix,q_batch in enumerate(output_strings):
                answers = self.qa_model.infer_batch(question_list=q_batch, text_list=[batch['c_text'][ix] for _ in range(len(q_batch))])

                this_scores = [f1(batch['a_text'][ix], ans) for ans in answers]

                if max(this_scores) > 0.75:
                    print(batch['c_text'][ix], "->", batch['q_text'][ix], batch['a_text'][ix])
                    print('***')
                    for j in range(len(q_batch)):
                        print(q_batch[j], "##", answers[j], this_scores[j])
                    print('***')
                    # exit()

                scores.append(this_scores)
            

            
            

            scores = torch.tensor(scores).to(self.device)

            sorted_scores, sorted_indices = torch.sort(scores, descending=True)
            
            sorted_seqs = torch.gather(candidates, 1, sorted_indices.unsqueeze(-1).expand(-1,-1, candidates.shape[2]))
            
            if top1:
                output = sorted_seqs[:, 0, :]
            else:
                output = sorted_seqs

            return output, torch.sum(output != BPE.pad_id, dim=-1), sorted_scores

            

        return None