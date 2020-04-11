import torch
import torch.nn as nn

from datasets.loaders import OOV_ID, PAD_ID


class RnnAqModel(nn.Module):
    def __init__(self, config, embeddings_init=None):
        super().__init__()
        self.config = config

        self.embeddings = nn.Embedding(
            config.prepro.vocab_size + 4, config.embedding_dim, padding_idx=PAD_ID
        ).cpu()  # TODO: this should come from a config
        if embeddings_init is not None:
            self.embeddings.weight = nn.Parameter(torch.from_numpy(embeddings_init))

        # Context encoder RNN
        self.context_encoder = nn.LSTM(
            input_size=config.embedding_dim + 3, hidden_size=512, num_layers=1, batch_first=True,
        )

        # Params for hidden state calc
        self.hidden_state_linear = nn.Linear(512, 512, bias=True)

    def forward(self, batch):
        max_ctxt_len = torch.max(batch["c_len"])
        curr_batch_size = batch["c"].size()[0]

        # force the context to fall within the embeddable space, then embed it
        context_coerced = torch.where(
            batch["c"] >= self.config.prepro.vocab_size + 4, torch.LongTensor([OOV_ID]), batch["c"]
        )
        context_embedded = self.embeddings(context_coerced)

        ans_pos_embedded = (
            torch.FloatTensor(curr_batch_size, max_ctxt_len, 3).zero_().scatter_(2, batch["a_pos"].unsqueeze(-1), 1)
        )

        context_augmented = torch.cat([context_embedded, ans_pos_embedded], 2)

        # Pack the seq, encode it, then pad it
        # NOTE: pack_padded requires sorted sequences - so sort the input, encode, then unsort...
        seq_lengths, perm_idx = batch["c_len"].sort(0, descending=True)
        context_augmented = context_augmented[perm_idx]
        context_augmented_packed = nn.utils.rnn.pack_padded_sequence(context_augmented, seq_lengths, batch_first=True)

        context_encoding, _ = self.context_encoder(
            context_augmented_packed, self.init_hidden_ctxt_encoder(curr_batch_size)
        )

        context_encoding, _ = nn.utils.rnn.pad_packed_sequence(context_encoding, batch_first=True)
        _, inv_idx = perm_idx.sort(0)
        context_encoding = context_encoding[inv_idx]

        # TODO: The rest of the condition encoding stuff :)

        # init state is tanh(Wh +b) where h is mean of context_encoding
        mean_ctxt_encoding = torch.sum(context_encoding, 1) / batch["c_len"].type(torch.FloatTensor).unsqueeze(-1)
        decoder_init_state = torch.tanh(self.hidden_state_linear(mean_ctxt_encoding))

        # Teacher forced decoder for training
        print(decoder_init_state.size())
        return batch["q"]

    def init_hidden_ctxt_encoder(self, batch_size):
        # the weights are of the form (nb_layers, batch_size, nb_lstm_units)
        hidden_a = torch.randn(1, batch_size, 512)
        hidden_b = torch.randn(1, batch_size, 512)

        # hidden_a = torch.autograd.Variable(hidden_a)
        # hidden_b = torch.autograd.Variable(hidden_b)

        return (hidden_a, hidden_b)
