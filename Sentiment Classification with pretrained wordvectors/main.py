import argparse
import random

import jieba
import numpy as np
import torch
import torch.autograd
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torchtext.data import Field, TabularDataset, BucketIterator
from torchtext.vocab import Vectors

from models.BiLSTM import BiLSTM
from models.BiLSTMAttn import BiLSTMAttn
from models.CNN import TextCNN


class Trainer(object):
    def __init__(self, model):
        self.model = model
        self.criterion = nn.NLLLoss()
        self.optimizer = optim.Adam(model.parameters(), args.learning_rate, weight_decay=1e-4)

    def train(self, text_iter):
        self.model.train()

        loss_list = []
        pred_list = []
        label_list = []
        for batch in text_iter:
            self.optimizer.zero_grad()

            outputs = self.model(batch.review)

            loss = self.criterion(outputs, batch.label)
            loss.backward()
            self.optimizer.step()

            loss_list.append(loss.item())
            pred_list.append(torch.argmax(outputs, dim=-1).cpu().numpy())
            label_list.append(batch.label.cpu().numpy())

        y_pred = np.concatenate(pred_list)
        y_true = np.concatenate(label_list)

        loss = np.mean(loss_list)
        acc = accuracy_score(y_true, y_pred)

        return loss, acc

    def evaluate(self, text_iter):
        self.model.eval()

        loss_list = []
        pred_list = []
        label_list = []
        with torch.no_grad():
            for batch in text_iter:
                outputs = self.model(batch.review)

                loss = self.criterion(outputs, batch.label)

                loss_list.append(loss.item())
                pred_list.append(torch.argmax(outputs, dim=-1).cpu().numpy())
                label_list.append(batch.label.cpu().numpy())

        y_pred = np.concatenate(pred_list)
        y_true = np.concatenate(label_list)

        loss = np.mean(loss_list)
        acc = accuracy_score(y_true, y_pred)

        return loss, acc

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-path', type=str, default='./dataset')
    parser.add_argument('-train_file', type=str, default='train.csv')
    parser.add_argument('-dev_file', type=str, default='dev.csv')
    parser.add_argument('-test_file', type=str, default='test.csv')
    parser.add_argument('-save_path', type=str, default='./model.pkl')
    parser.add_argument('-model', type=str, default="bilstm_attn", help="[cnn, bilstm, bilstm_attn]")

    parser.add_argument('-batch_size', type=int, default=16)
    parser.add_argument('-embedding_size', type=int, default=300)
    parser.add_argument('-hidden_size', type=int, default=64)
    parser.add_argument('-learning_rate', type=float, default=1e-4)
    parser.add_argument('-dropout', type=float, default=0.5)
    parser.add_argument('-epochs', type=int, default=30)

    parser.add_argument('-seed', type=int, default=1)
    args = parser.parse_args()

    print(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    print("Loading Data...")
    LABEL = Field(sequential=False, use_vocab=False, batch_first=True)
    REVIEW = Field(sequential=True, use_vocab=True, tokenize=lambda x: jieba.lcut(x), batch_first=True)

    fields = [('label', LABEL), ('review', REVIEW)]

    train, dev, test = TabularDataset.splits(
        path=args.path,
        train=args.train_file,
        validation=args.dev_file,
        test=args.test_file,
        format='csv',
        fields=fields,
        skip_header=True
    )

    # print(vars(train[0]))
    zhihu = Vectors(name='sgns.zhihu.word')
    REVIEW.build_vocab(train, vectors=zhihu)
    # U, S, V = torch.pca_lowrank(REVIEW.vocab.vectors, q=64)
    # vec = torch.matmul(REVIEW.vocab.vectors, V)

    train_iter, dev_iter, test_iter = BucketIterator.splits(
        datasets=(train, dev, test),
        batch_size=args.batch_size,
        device='cuda',
        sort=False
    )

    print("Building Model...")
    if args.model == "cnn":
        model = TextCNN(vocab_size=len(REVIEW.vocab),
                        embedding_size=args.embedding_size,
                        hidden_size=args.hidden_size,
                        filter_sizes=[3, 4, 5],
                        dropout=args.dropout,
                        pretrain_wordvector=REVIEW.vocab.vectors)
    elif args.model == "bilstm":
        model = BiLSTM(vocab_size=len(REVIEW.vocab),
                       embedding_size=args.embedding_size,
                       hidden_size=args.hidden_size,
                       dropout=args.dropout,
                       pretrained_wordvector=REVIEW.vocab.vectors)
    elif args.model == "bilstm_attn":
        model = BiLSTMAttn(vocab_size=len(REVIEW.vocab),
                           embedding_size=args.embedding_size,
                           hidden_size=args.hidden_size,
                           dropout=args.dropout,
                           pretrained_wordvector=REVIEW.vocab.vectors)
    else:
        raise ValueError("Model should be cnn, bilstm or bilstm_attn, {} is invalid.".format(args.model))

    if torch.cuda.is_available():
        model = model.cuda()

    trainer = Trainer(model)

    best_acc = 0
    train_list = []
    dev_list = []
    for i in range(args.epochs):
        print("Epoch: {} ################################".format(i))
        train_loss, train_acc = trainer.train(train_iter)
        train_list.append(train_loss)
        dev_loss, dev_acc = trainer.evaluate(dev_iter)
        dev_list.append(dev_loss)
        print("Train Loss: {:.4f} Acc: {:.4f}".format(train_loss, train_acc))
        print("Dev   Loss: {:.4f} Acc: {:.4f}".format(dev_loss, dev_acc))
        if dev_acc > best_acc:
            best_acc = dev_acc
            trainer.save(args.save_path)
        print("###########################################")
    trainer.load(args.save_path)
    test_loss, test_acc = trainer.evaluate(test_iter)
    print("Test   Loss: {:.4f} Acc: {:.4f}".format(test_loss, test_acc))

    import matplotlib.pyplot as plt

    x_values = list(range(0, args.epochs))
    plt.xticks(x_values)
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.plot(x_values, train_list, label="train loss", color="#F08080")
    plt.plot(x_values, dev_list, label="dev loss", color="#0B7093")
    plt.legend()
    plt.savefig("train-dev-loss.png")
    plt.show()
