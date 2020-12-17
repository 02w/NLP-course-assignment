import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, dropout, tag_size):
        super(Model, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(vocab_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, batch_first=True, bidirectional=True)
        self.decoder = nn.Linear(hidden_size * 2, tag_size)
        self.init_weights()

    def init_weights(self):
        init_range = 0.1
        self.encoder.weight.data.uniform_(-init_range, init_range)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_range, init_range)

    def forward(self, x):
        emb = self.drop(self.encoder(x))  # (B, L) -> (B, L, D)
        output, (_, _) = self.rnn(emb)        # output: (B, L, 2 * D)
        output = self.drop(output)

        # hn = torch.cat(torch.unbind(hn, dim=0), dim=1)  # (2, B, D) -> (B, 2 * D)
        decoded = self.decoder(output)                      # (B, L, 2 * D) -> (B, L, T)
        return decoded
