import argparse
import random

import numpy as np
import torch
import torch.autograd
import torch.optim as optim
from torch.utils.data import DataLoader

import utils
from evaluation import eval_metric
from models import BiLSTM_CRF_batch


class Trainer(object):
    def __init__(self, model):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), args.learning_rate)

    def run(self, data_loader, train=True):
        if train:
            self.model.train()
        else:
            self.model.eval()

        loss_list = []
        pred_list = []
        label_list = []
        inputs_list = []
        len_list = []

        for labels, inputs, lengths in data_loader:
            self.optimizer.zero_grad()

            _, outputs = self.model(inputs, lengths)

            loss = model.neg_log_likelihood(inputs, labels, lengths)

            if train:
                loss.backward()
                self.optimizer.step()

            loss_list.append(loss.item())
            pred_list.append([i.cpu().numpy() for i in outputs])
            label_list.append(labels.cpu().numpy())
            inputs = inputs.cpu().numpy()
            len_list.append(lengths)
            inputs_list.append(inputs)

        loss_val = np.mean(loss_list)

        acc_list = []
        p_list = []
        r_list = []
        f1_list = []
        for sent, gold, pred, sent_len in zip(inputs_list, label_list, pred_list, len_list):
            acc, p, r, f1 = eval_metric(sent, gold, pred, sent_len)
            acc_list.append(acc)
            p_list.append(p)
            r_list.append(r)
            f1_list.append(f1)

        acc_val = np.mean(acc_list)
        p_val = np.mean(p_list)
        r_val = np.mean(r_list)
        f1_val = np.mean(f1_list)

        return loss_val, acc_val, p_val, r_val, f1_val

    # def evaluate(self, data_loader):
    #     self.model.eval()
    #
    #     loss_list = []
    #     pred_list = []
    #     label_list = []
    #     with torch.no_grad():
    #         for labels, inputs in data_loader:
    #             outputs = self.model(inputs)
    #
    #             loss = self.criterion(outputs, labels)
    #
    #             loss_list.append(loss.item())
    #             pred_list.append(torch.argmax(outputs, dim=-1).cpu().numpy())
    #             label_list.append(labels.cpu().numpy())
    #
    #     y_pred = np.concatenate(pred_list)
    #     y_true = np.concatenate(label_list)
    #
    #     loss = np.mean(loss_list)
    #     acc = accuracy_score(y_true, y_pred)
    #
    #     return loss, acc

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('-f')
    parser.add_argument('-train_file', type=str, default='./conll04/conll04_train.json')
    parser.add_argument('-dev_file', type=str, default='./conll04/conll04_dev.json')
    parser.add_argument('-test_file', type=str, default='./conll04/conll04_test.json')
    parser.add_argument('-save_path', type=str, default='./model.pkl')
    parser.add_argument('-model', type=str, default="bilstm_crf", help="bilstm_crf")

    parser.add_argument('-batch_size', type=int, default=16)
    parser.add_argument('-embedding_size', type=int, default=256)
    parser.add_argument('-hidden_size', type=int, default=128)
    parser.add_argument('-learning_rate', type=float, default=1e-3)
    parser.add_argument('-dropout', type=float, default=0.5)
    parser.add_argument('-epochs', type=int, default=60)

    parser.add_argument('-seed', type=int, default=1)
    args = parser.parse_args()

    print(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    print("Loading Data...")

    dataests, vocab, type2label = utils.build_dataset(args.train_file,
                                                      args.dev_file,
                                                      args.test_file)

    train_loader, dev_loader, test_loader = (
        DataLoader(dataset=dataset,
                   batch_size=args.batch_size,
                   collate_fn=utils.collate_fn,
                   shuffle=dataset.train)
        for i, dataset in enumerate(dataests))

    print("Building Model...")
    if args.model == "bilstm_crf":
        model = BiLSTM_CRF_batch.Model(vocab_size=len(vocab),
                                       tag_to_ix=type2label,
                                       embedding_dim=args.embedding_size,
                                       hidden_dim=args.hidden_size,
                                       dropout=args.dropout,
                                       )
    else:
        raise ValueError("Model should be bilstm_crf, {} is invalid.".format(args.model))

    if torch.cuda.is_available():
        model = model.cuda()

    trainer = Trainer(model)

    best_f1 = 0
    train_list = []
    dev_list = []
    for i in range(args.epochs):
        print("Epoch: {} ################################".format(i))
        train_loss, train_acc, train_p, train_r, train_f1 = trainer.run(train_loader)
        train_list.append(train_loss)
        dev_loss, dev_acc, dev_p, dev_r, dev_f1 = trainer.run(dev_loader, train=False)
        dev_list.append(dev_loss)
        print("Train Loss: {:.4f} Acc: {:.4f} P: {:.4f} R: {:.4f} F1: {:.4f}".format(train_loss, train_acc, train_p, train_r, train_f1))
        print("Dev   Loss: {:.4f} Acc: {:.4f} P: {:.4f} R: {:.4f} F1: {:.4f}".format(dev_loss, dev_acc, dev_p, dev_r, dev_f1))
        if dev_f1 > best_f1:
            best_f1 = dev_f1
            trainer.save(args.save_path)
        print("###########################################")
    trainer.load(args.save_path)
    test_loss, test_acc, test_p, test_r, test_f1 = trainer.run(test_loader, train=False)
    print("Test   Loss: {:.4f} Acc: {:.4f} P: {:.4f} R: {:.4f} F1: {:.4f}".format(test_loss, test_acc, test_p, test_r, test_f1))

    import matplotlib.pyplot as plt

    x_values = list(range(0, args.epochs))
    plt.xticks(range(0, args.epochs, 2))
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.plot(x_values, train_list, label="train loss", color="#F08080")
    plt.plot(x_values, dev_list, label="dev loss", color="#0B7093")
    plt.legend()
    plt.savefig("train-dev-loss.png")
    plt.show()

