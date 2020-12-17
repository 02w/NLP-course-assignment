import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLSTMAttn(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, dropout, pretrained_wordvector=None):
        super(BiLSTMAttn, self).__init__()
        self.hidden_size = hidden_size
        self.drop = nn.Dropout(dropout)
        if pretrained_wordvector is None:
            self.encoder = nn.Embedding(vocab_size, embedding_size)
        else:
            self.encoder = nn.Embedding.from_pretrained(pretrained_wordvector)
            self.encoder.requires_grad_(True)
            embedding_size = pretrained_wordvector.size(1)
        self.rnn = nn.LSTM(embedding_size, hidden_size, batch_first=True, bidirectional=True)
        self.decoder = nn.Linear(hidden_size * 2, 2)
        self.init_weights()

    def init_weights(self):
        init_range = 0.1
        self.encoder.weight.data.uniform_(-init_range, init_range)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_range, init_range)

    def attention(self, lstm_output, hn):
        hidden = hn.view(-1, self.hidden_size * 2, 1)
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)
        softmax_weights = F.softmax(attn_weights, dim=1)
        context = torch.bmm(lstm_output.transpose(1, 2), softmax_weights.unsqueeze(2)).squeeze(2)
        return context, softmax_weights

    def forward(self, x):
        emb = self.drop(self.encoder(x))  # (B, L) -> (B, L, D)
        output, (hn, _) = self.rnn(emb)  # hn: (2, B, D)

        attn_output, attn = self.attention(output, hn)

        decoded = self.decoder(self.drop(attn_output))  # (B, 2 * D) -> (B, 2)

        out = F.log_softmax(decoded, dim=-1)
        # softmax -> (0, 1)   log(softmax) -> (-inf, 1)  NLLLoss() -> (0, 1)
        # (B, 2): 2维分别相当于该样本为0的概率和该样本为1的概率
        return out
