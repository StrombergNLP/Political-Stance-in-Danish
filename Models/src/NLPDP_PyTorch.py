import sys
import collections
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
from torch.nn.utils.rnn import pack_sequence
import torch.nn.functional as F
import torch.optim as optim
import sklearn.metrics as sk

traindatafile = "../data/preprocess-output/train.txt"
testdatafile = "../data/preprocess-output/test.txt"
devdatafile = "../data/preprocess-output/dev.txt"

LABELS = ['S', 'D', 'Q', 'C']


# Source of inspiration:
# https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html#sphx-glr-beginner-nlp-sequence-models-tutorial-py
class BranchLSTM(nn.Module):
    def __init__(self, emb_dim, lstm_num, lstm_dim, relu_num, relu_dim, num_labels):
        super(BranchLSTM, self).__init__()
        self.lstm_dim = lstm_dim
        self.lstm_num = lstm_num
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality lstm_dim.
        self.lstm = nn.LSTM(emb_dim, lstm_dim, lstm_num)
        self.hidden = self.init_hidden()

        # The linear layer(s) that maps from hidden state space to label space
        dense_layers = collections.OrderedDict()
        dense_layers["lin0"] = torch.nn.Linear(lstm_dim, relu_dim)
        dense_layers["rec0"] = torch.nn.ReLU()
        for i in range(relu_num - 1):
            dense_layers["lin%d" % (i + 1)] = torch.nn.Linear(relu_dim, relu_dim)
            dense_layers["rec%d" % (i + 1)] = torch.nn.ReLU()
        dense_layers["drop"] = torch.nn.Dropout(p=0.5)
        dense_layers["lin%d" % relu_num] = torch.nn.Linear(relu_dim, num_labels)
        # dense_layers["sm"] = torch.nn.LogSoftmax(dim=1)
        self.hidden2label = torch.nn.Sequential(dense_layers)
        # self.hidden2label = nn.Linear(lstm_dim, num_labels)

    def init_hidden(self):  # (h_0, c_0)
        # Before we've done anything, we dont have any hidden state.
        # The axes semantics are (num_layers, minibatch_size, lstm_dim)
        return (torch.zeros(self.lstm_num, 1, self.lstm_dim),
                torch.zeros(self.lstm_num, 1, self.lstm_dim))

    def forward(self, branch):
        lstm_out, self.hidden = self.lstm(
            branch.view(len(branch), 1, -1),
            self.hidden
        )
        label_space = self.hidden2label(lstm_out.view(len(branch), -1))
        score = F.log_softmax(label_space, dim=1)
        # score = F.log_softmax(lstm_out.view(len(branch), -1), dim=1)
        return score


def train(training_data, model, loss_func, optimizer, epochs):
    epoch_loss = 0.0
    # dev_acc = 0.0
    for epoch in range(epochs):
        # print("Epoch", epoch)
        # running_loss = 0.0
        masks = set()  # set of already seen tweet ids
        for i, branch in enumerate(training_data):
            # Step 1. Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Also, we need to clear out the hidden state of the LSTM,
            # detaching it from its history on the last instance.
            model.hidden = model.init_hidden()

            # Step 2. Get our inputs ready for the network
            branch_vecs = []
            branch_labels = []
            exclude = []
            for index, (t_id, label, vec) in enumerate(branch):
                if t_id in masks:  # exclude already seen tweets
                    exclude.append(index)
                else:
                    masks.add(t_id)
                branch_vecs.append(vec)
                branch_labels.append(label)
            if not branch_vecs:
                continue

            inputs = torch.tensor(branch_vecs)
            label_scores = model(inputs)
            targets = torch.tensor(branch_labels)

            if exclude:  # exclude repeating tweets from loss function
                indices = torch.LongTensor(exclude)
                targets = torch.index_select(targets, 0, indices)
                label_scores = torch.index_select(label_scores, 0, indices)

            loss = loss_func(label_scores, targets)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            # if (i % 100 == 0):
            #     print('[%d, %5d] loss: %.3f' %
            #         (epoch + 1, i, running_loss / 1000))
            #     running_loss = 0.0
        print('Epoch %d, loss: %.5f' % (epoch + 1, epoch_loss / 1000))
        epoch_loss = 0
        # _, acc, _ = test(dev_data, model)
        # if acc < dev_acc:
        #     break # early stop if accuracy falls for dev set
        # else:
        #     dev_acc = acc


# Source of inspiration:
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
def test(test_data, model):
    labels_true = []
    labels_pred = []
    # Here we don't need to train, so the code is wrapped in torch.no_grad()
    with torch.no_grad():
        for branch in test_data:
            # First prepare data
            branch_vecs = []
            for _, label, vec in branch:
                branch_vecs.append(vec)
                labels_true.append(label)  # store true labels
            inputs = torch.tensor(branch_vecs)
            # Run it through the model
            outputs = model(inputs)
            # Compute and store indices of the max values
            predicted = torch.argmax(outputs.data, dim=1)
            labels_pred.extend(predicted.numpy())

    # Statistics
    cm = sk.confusion_matrix(labels_true, labels_pred, labels=[0, 1, 2, 3])
    print("Confusion matrix:")
    print(cm)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sdqc_acc = cm.diagonal()
    acc = sk.accuracy_score(labels_true, labels_pred)
    f1 = sk.f1_score(labels_true, labels_pred, average='macro')
    print("SDQC acc:", sdqc_acc)
    print("Accuracy: %.5f" % acc)
    print("F1-macro:", f1)
    return sdqc_acc, acc, f1


def loadInstances(filename):
    """Return a list of branch lists in the format:
    [[branch_0], ..., [branch_n]] and each instance within
    a branch has then the format: (0:id, 1:label, 2:vector)"""
    max_emb = 0
    instances = []
    with open(filename, 'r') as f:
        branches = f.read().split("\n\n")  # lines of vectors ([1:] skip header)
        for branch in branches:
            lines = branch.split("\n")  # each individual line
            branch_instance = []  # renew for each branch
            for line in lines:  # go through lines and append instances to a branch
                if line:
                    instance = line.strip().split('\t')
                    instance_id = int(instance[0])
                    instance_label = int(instance[1])
                    values = instance[2].strip("[").strip("]").split(',')
                    instance_vec = [float(i.strip()) for i in values]
                    max_emb = max(max_emb, len(instance_vec))
                    branch_instance.append((instance_id, instance_label, instance_vec))
            if len(branch_instance) > 0:
                instances.append(branch_instance)  # all branch instances
    return (instances, max_emb)


training_data, emb_size = loadInstances(traindatafile)
dev_data, _ = loadInstances(devdatafile)
training_data.extend(dev_data)  # combine train and dev
test_data, _ = loadInstances(testdatafile)

EMB = emb_size
LSTM_LAYERS = [1, 2]
LSTM_DIM = [100, 200, 300]
RELU_LAYERS = [1, 2, 3, 4]
RELU_DIM = [100, 200, 300, 400, 500]
L2 = [0.0, 1e-4, 3e-4, 1e-3]

EPOCHS = [30, 50, 70, 100]

model = BranchLSTM(EMB, LSTM_LAYERS[1], LSTM_DIM[0], RELU_LAYERS[1], RELU_DIM[4], len(LABELS))
loss_func = nn.NLLLoss()  # nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=L2[0])  # optim.Adadelta(model.parameters(), lr=0.1)

train(training_data, model, loss_func, optimizer, EPOCHS[0])
test(test_data, model)


def run_test(epochs, lstm_layers, lstm_units, relu_layers, relu_units, l2_reg):
    training_data, emb_size = loadInstances(traindatafile)
    dev_data, _ = loadInstances(devdatafile)
    training_data.extend(dev_data)  # combine train and dev
    test_data, _ = loadInstances(testdatafile)
    model = BranchLSTM(emb_size, lstm_layers, lstm_units, relu_layers, relu_units, len(LABELS))
    loss_func = nn.NLLLoss()  # nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001,
                          weight_decay=l2_reg)  # optim.Adadelta(model.parameters(), lr=0.1)

    train(training_data, model, loss_func, optimizer, epochs)
    test_res = test(test_data, model)
    return test_res
