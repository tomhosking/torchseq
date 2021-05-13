import torch
import copy

from torchseq.utils.functions import onehot


class MLPClassifier(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_heads):
        super(MLPClassifier, self).__init__()
        self.linear = torch.nn.Linear(input_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = torch.nn.Linear(hidden_dim, output_dim * num_heads)

        self.drop1 = torch.nn.Dropout(p=0.2)
        self.drop2 = torch.nn.Dropout(p=0.2)
        self.num_heads = num_heads
        self.output_dim = output_dim

    def forward(self, x):
        outputs = self.drop1(torch.nn.functional.relu(self.linear(x)))
        outputs = self.drop2(torch.nn.functional.relu(self.linear2(outputs)))
        outputs = self.linear3(outputs)
        return outputs.reshape(-1, self.num_heads, self.output_dim)


class VQCodePredictor(torch.nn.Module):
    def __init__(self, config, transitions=None):
        super(VQCodePredictor, self).__init__()

        self.classifier = MLPClassifier(config.input_dim, config.output_dim, config.hidden_dim, config.num_heads)

        self.transitions = transitions
        self.config = config

        # self.criterion = torch.nn.CrossEntropyLoss().cuda() # computes softmax and then the cross entropy

        self.optimizer = torch.optim.Adam(self.classifier.parameters(), lr=config.lr)

    def infer(self, encoding):
        # TODO: Batchify this...
        self.classifier.eval()
        outputs = self.classifier(encoding)
        all_pred_codes = []
        for logits in outputs:
            joint_probs = [([], 0)]
            for h_ix in range(self.config.num_heads):
                new_hypotheses = []
                for i, (combo, prob) in enumerate(joint_probs):
                    if h_ix > 0 and self.transitions is not None:
                        prev_oh = onehot(torch.tensor(combo[-1]).to(logits.device), N=self.config.output_dim) * 1.0
                        curr_logits = logits[h_ix, :] + self.transitions[h_ix - 1](prev_oh)
                    else:
                        curr_logits = logits[h_ix, :]
                    probs, predicted = torch.topk(torch.softmax(curr_logits, -1), 3, -1)
                    for k in range(2):
                        new_hyp = [copy.copy(combo), copy.copy(prob)]
                        new_hyp[0].append(predicted[k].item())
                        new_hyp[1] += torch.log(probs[k]).item()

                        new_hypotheses.append(new_hyp)

                joint_probs = new_hypotheses
                joint_probs = sorted(joint_probs, key=lambda x: x[1], reverse=True)[:3]
            pred_codes = [x[0] for x in sorted(joint_probs, key=lambda x: x[1], reverse=True)[:2]]
            all_pred_codes.append(pred_codes)

        # HACK: return top-1 for now
        top_1_codes = torch.IntTensor(all_pred_codes)[:, 0, :].to(encoding.device)
        return top_1_codes

    def train_step(self, encoding, code_mask):
        # Encoding should be shape: bsz x dim
        # code_mask should be a n-hot vector, shape: bsz x codebook
        self.classifier.train()

        self.optimizer.zero_grad()
        outputs = self.classifier(encoding)

        logits = [outputs[:, 0, :].unsqueeze(1)]

        # Use teacher forcing to train the subsequent heads
        for head_ix in range(1, self.config.num_heads):
            if self.transitions is not None:
                logits.append(
                    outputs[:, head_ix, :].unsqueeze(1)
                    + self.transitions[head_ix - 1](code_mask[:, head_ix - 1, :]).detach().unsqueeze(1)
                )
            else:
                logits.append(outputs[:, head_ix, :].unsqueeze(1))
        logits = torch.cat(logits, dim=1)

        loss = torch.sum(
            -1 * torch.nn.functional.log_softmax(logits, dim=-1) * code_mask / code_mask.sum(dim=-1, keepdims=True),
            dim=-1,
        ).mean()  #
        loss.backward()
        self.optimizer.step()

        return loss.detach().item()

    def forward(self, encoding):
        logits = self.classifier(encoding)

        return logits
