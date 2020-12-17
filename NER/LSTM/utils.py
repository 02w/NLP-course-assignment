import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import json


def mask_matrix(lengths, tag_size):
    cre = torch.zeros((len(lengths), max(lengths))).long()
    cre_matrix = torch.ones((len(lengths), max(lengths), tag_size)).long()
    if torch.cuda.is_available():
        cre = cre.cuda()
        cre_matrix = cre_matrix.cuda()
    for i, lens in enumerate(lengths):
        one = torch.zeros(tag_size)
        one[-1] = 1
        cre[i][:lens] = 1
        cre_matrix[i][lens - 1:] = one

    return cre, cre_matrix


class Loss(object):
    @staticmethod
    def lstm_loss(logits, targets, lengths):
        """计算损失
        参数:
            logits: [B, L, out_size]
            targets: [B, L]
            lengths: [B]
        """
        # PAD = 10
        # assert PAD is not None

        # mask = (targets != PAD)  # [B, L]
        mask, _ = mask_matrix(lengths, logits.size()[2])
        mask = mask.bool()
        targets = targets[mask]
        out_size = logits.size(2)
        logits = logits.masked_select(
            mask.unsqueeze(2).expand(-1, -1, out_size)
        ).contiguous().view(-1, out_size)

        assert logits.size(0) == targets.size(0)
        loss = F.cross_entropy(logits, targets)

        return loss


class DataParser(object):
    def __init__(self, file):
        self.file = file
        self.type2label = {'B-Loc': 0, 'I-Loc': 1, 'B-Org': 2, 'I-Org': 3, 'B-Peop': 4, 'I-Peop': 5, 'B-Other': 6, 'I-Other': 7, 'O': 8}
        self.tokens = []
        self.labels = []

    def parse(self):
        with open(self.file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for sample in data:
            self.tokens.append(sample['tokens'])
            label = [self.type2label['O'] for i in range(len(sample['tokens']))]
            for entity in sample['entities']:
                label[entity['start']] = self.type2label['B-' + entity['type']]
                for i in range(entity['start'] + 1, entity['end']):
                    label[i] = self.type2label['I-' + entity['type']]
            self.labels.append(label)


class Vocabulary(object):
    PAD = '<PAD>'
    UNK = '<UNK>'

    def __init__(self):
        self.token2id = {self.PAD: 0, self.UNK: 1}
        self.id2token = {0: self.PAD, 1: self.UNK}

    def add_token(self, token):
        if token not in self.token2id:
            self.token2id[token] = len(self.token2id)
            self.id2token[self.token2id[token]] = token

    def __len__(self):
        return len(self.token2id)

    def encode(self, text):
        return [self.token2id.get(x, self.token2id[self.UNK]) for x in text]

    def decode(self, ids):
        return [self.id2token.get(x) for x in ids]


class MyDataset(Dataset):
    def __init__(self, labels, inputs, lengths, train=True):
        self.labels = labels
        self.inputs = inputs
        self.train = train
        self.lengths = lengths

    def __getitem__(self, item):
        return torch.LongTensor(self.labels[item]), \
               torch.LongTensor(self.inputs[item]), \
               self.lengths[item]

    def __len__(self):
        return len(self.inputs)


def collate_fn(data):
    labels, inputs, lengths = map(list, zip(*data))
    # labels = torch.cat(labels, dim=0)
    inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=0)

    if torch.cuda.is_available():
        labels = labels.cuda()
        inputs = inputs.cuda()

    return labels, inputs, lengths


def build_dataset(train_path, dev_path, test_path):
    vocab = Vocabulary()

    def load_data(path, train=True):
        data = DataParser(path)
        data.parse()
        # labels = []
        inputs = []
        lengths = []
        # for text in data[1:]:
        #     text = text.strip()
        #     label, tokens = [int(text[0])], jieba.lcut(text[2:])
        for tokens in data.tokens:
            if train:
                for token in tokens:
                    vocab.add_token(token)

            tokens = vocab.encode(tokens)
            # labels.append(label)
            inputs.append(tokens)
            lengths.append(len(tokens))
        return data.labels, inputs, lengths

    train_dataset = MyDataset(*load_data(train_path))
    dev_dataset = MyDataset(*load_data(dev_path), train=False)
    test_dataset = MyDataset(*load_data(test_path), train=False)
    type2label = DataParser(train_path).type2label
    return (train_dataset, dev_dataset, test_dataset), vocab, type2label
