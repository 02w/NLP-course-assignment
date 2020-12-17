import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, filter_sizes, dropout, pretrain_wordvector=None):
        super(TextCNN, self).__init__()
        if pretrain_wordvector is None:
            self.embedding = nn.Embedding(vocab_size, embedding_size)
        else:
            self.embedding = nn.Embedding.from_pretrained(pretrain_wordvector)
            self.embedding.requires_grad_(True)
            embedding_size = pretrain_wordvector.size(1)
        self.convs = nn.ModuleList(
            [nn.Conv1d(embedding_size, hidden_size, k) for k in filter_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * len(filter_sizes), 2)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x))  # (B, D, L) -> (B, H, L)
        x = F.max_pool1d(x, x.size(2)).squeeze()  # (B, H, L) -> (B, H)
        return x

    def forward(self, x):
        emb = self.embedding(x)  # (B, L) -> (B, L, D)
        emb = self.dropout(emb)
        emb = emb.transpose(1, 2)  # (B, L, D) -> (B, D, L)
        out = torch.cat([self.conv_and_pool(emb, conv) for conv in self.convs], -1)  # 3 * (B, D, L) -> (B, 3 * H)

        out = self.fc(out)  # (B, 3 * H) -> (B, 2)
        out = F.log_softmax(out, dim=-1)
        return out
