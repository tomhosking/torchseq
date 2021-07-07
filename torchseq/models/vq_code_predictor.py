import torch
import torch.nn as nn
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

    def forward(self, x, seq=None):
        outputs = self.drop1(torch.nn.functional.relu(self.linear(x)))
        outputs = self.drop2(torch.nn.functional.relu(self.linear2(outputs)))
        outputs = self.linear3(outputs)
        return outputs.reshape(-1, self.num_heads, self.output_dim)


class LstmClassifier(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_heads, seq_dim=None):
        super(LstmClassifier, self).__init__()
        self.lstm_in = torch.nn.Linear(input_dim, hidden_dim)
        self.rnn = nn.LSTMCell(
            hidden_dim,
            hidden_dim,
        )
        self.lstm_out = torch.nn.Linear(hidden_dim, output_dim)

        if seq_dim is not None and seq_dim != hidden_dim:
            self.seq_proj = torch.nn.Linear(seq_dim, hidden_dim, bias=False)
        else:
            self.seq_proj = None

        self.drop1 = torch.nn.Dropout(p=0.2)
        self.drop2 = torch.nn.Dropout(p=0.2)
        self.num_heads = num_heads
        self.output_dim = output_dim

    def forward(self, x, seq=None):
        outputs = self.drop1(torch.nn.functional.relu(self.lstm_in(x)))
        rnn_out = []
        hx, cx = outputs, torch.zeros_like(outputs)
        if self.seq_proj is not None and seq is not None:
            seq = self.seq_proj(seq)
        for hix in range(self.num_heads if seq is None else seq.shape[1]):
            hx, cx = self.rnn(seq[:, hix, :] if seq is not None else hx, (hx, cx))
            rnn_out.append(hx)
        outputs = torch.stack(rnn_out, dim=1)
        outputs = self.lstm_out(outputs)
        return outputs


class VQCodePredictor(torch.nn.Module):
    def __init__(self, config, transitions=None, embeddings=None):
        super(VQCodePredictor, self).__init__()

        if config.get("use_lstm", False):
            self.classifier = LstmClassifier(
                config.input_dim,
                config.output_dim,
                config.hidden_dim,
                config.num_heads,
                seq_dim=config.get("lstm_seq_dim", None),
            )
        else:
            self.classifier = MLPClassifier(config.input_dim, config.output_dim, config.hidden_dim, config.num_heads)

        self.transitions = transitions

        self.embeddings = embeddings

        self.config = config

        # self.criterion = torch.nn.CrossEntropyLoss().cuda() # computes softmax and then the cross entropy

        self.optimizer = torch.optim.Adam(self.classifier.parameters(), lr=config.lr)

    def infer(self, encoding):
        # TODO: Batchify this...
        self.classifier.eval()
        outputs = self.classifier(encoding)
        all_pred_codes = []
        for bix, logits in enumerate(outputs):
            joint_probs = [([], 0)]
            for h_ix in range(self.config.num_heads):
                new_hypotheses = []
                for i, (combo, prob) in enumerate(joint_probs):
                    if h_ix > 0 and self.transitions is not None:
                        prev_oh = onehot(torch.tensor(combo[-1]).to(logits.device), N=self.config.output_dim) * 1.0
                        curr_logits = logits[h_ix, :] + self.transitions[h_ix - 1](prev_oh).detach()
                    elif self.config.get("autoregressive_lstm", False):
                        seq_dim = (
                            self.config.hidden_dim
                            if self.config.get("lstm_seq_dim", None) is None
                            else self.config.get("lstm_seq_dim", None)
                        )
                        seq_init = torch.zeros(1, 1, seq_dim).to(encoding.device)
                        seq_embedded = [
                            embed(torch.tensor(x).to(encoding.device)).unsqueeze(0).unsqueeze(1).detach()
                            for x, embed in zip(combo, self.embeddings)
                        ]
                        seq = torch.cat([seq_init, *seq_embedded], dim=1)
                        curr_logits = self.classifier(encoding[bix].unsqueeze(0), seq)[0, h_ix, :]
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

    def train_step(self, encoding, code_mask, take_step=True):
        # Encoding should be shape: bsz x dim
        # code_mask should be a n-hot vector, shape: bsz x codebook
        self.classifier.train()

        self.optimizer.zero_grad()
        if self.config.get("autoregressive_lstm", False):
            seq_dim = (
                self.config.hidden_dim
                if self.config.get("lstm_seq_dim", None) is None
                else self.config.get("lstm_seq_dim", None)
            )
            seq_init = torch.zeros(encoding.shape[0], 1, seq_dim).to(encoding.device)
            embedded_seq = [
                torch.matmul(code_mask[:, hix, :].float(), self.embeddings[hix].weight.detach()).unsqueeze(1)
                for hix in range(self.config.num_heads - 1)
            ]

            seq = torch.cat([seq_init, *embedded_seq], dim=1)
        else:
            seq = None
        outputs = self.classifier(encoding, seq=seq)

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
        if take_step:
            loss.backward()
            self.optimizer.step()

        return loss.detach()

    def forward(self, encoding):
        raise Exception(
            "fwd() shouldn't be called on the code predictor! Use either the training or inference methods"
        )
        logits = self.classifier(encoding)

        return logits
