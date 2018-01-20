import csv
import numpy as np
import pandas as pd

# scikit-learn
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as O
from torch.autograd import Variable

####################
# Training
####################
# Read csv file
data = pd.read_csv("train.csv")

# Change string to numerical in sex field
data = data.replace("male", 1).replace("female", 0)
# Remove some nonvaluable field
data = data.drop(["Name", "Ticket", "Embarked", "Cabin", "Fare"], axis=1)
# Remove missing value
data = data.dropna()
# Split dataset into training set and test one
X = data.values[:, 2:]
Y = data.values[:, 1].astype(dtype=np.int64)
train_x, test_x, train_t, test_t = train_test_split(X, Y, test_size=0.1)

# Naive bayes
gnb = GaussianNB()
gnb.fit(train_x, train_t)

result = gnb.predict(test_x)
num_right = np.sum(result == test_t)
print("Naive bayes")
print("Accuracy {:.2f}\n".format(num_right / len(test_t)))

# Support Vector Machine
clf = SVC()
clf.fit(train_x, train_t)

result = clf.predict(test_x)
num_right = np.sum(result == test_t)
print("Support Vector Machine")
print("Accuracy {:.2f}\n".format(num_right / len(test_t)))

# Neural Network
class NeuralNetwork(nn.Module):
    def __init__(self, n_in, n_hidden, n_out):
        super(NeuralNetwork, self).__init__()
        self.input = nn.Linear(n_in, n_hidden)
        self.output = nn.Linear(n_hidden, n_out)

    def forward(self, x):
        h = F.sigmoid(self.input(x))
        y = F.sigmoid(self.output(h))
        return y


batchsize = 50
epochs = 4000
learning_rate = 0.01
n_batch = len(train_x) // batchsize
n_in = len(train_x[0])
n_hidden = 3
n_out = 2

network = NeuralNetwork(n_in, n_hidden, n_out)
criterion = nn.CrossEntropyLoss()
optimizer = O.Adam(network.parameters(), lr=learning_rate)

for epoch in range(epochs):
    if epoch % 100 == 0:
        print("Epoch {}".format(epoch))
    train_x, train_t = shuffle(train_x, train_t)
    # Mini batch learning
    for i in range(n_batch):
        start = i * batchsize
        end = start + batchsize
        x_var = Variable(torch.FloatTensor(train_x[start:end]))
        t_var = Variable(torch.LongTensor(train_t[start:end]))
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        y_var = network(x_var)
        loss = criterion(y_var, t_var)
        loss.backward()
        optimizer.step()

# Test the model
test_var = Variable(torch.FloatTensor(test_x), volatile=True)
result = network(test_var)
values, labels = torch.max(result, 1)
num_right = np.sum(test_t == labels.data.numpy())
print("Neural Network")
print("Accuracy {:.2f}\n".format(num_right / len(test_t)))

####################
# Test
####################
path = "test.csv"
data = pd.read_csv(path)

# Change string to numerical in sex field
data = data.replace("male", 1).replace("female", 0)
# Remove some nonvaluable field
data = data.drop(["Name", "Ticket", "Embarked", "Cabin", "Fare"], axis=1)
# Give median values to NA in age field
data["Age"].fillna(data.Age.median(), inplace=True)
test_set = data.values[:, 1:]
test_var = Variable(torch.FloatTensor(test_set), volatile=True)
result = network(test_var)
values, labels = torch.max(result, 1)
with open("result.csv", "w") as f:
    writer = csv.writer(f, lineterminator="\n")
    writer.writerow(["PassengerId", "Survived"])
    for pid, survived in zip(data.values[:, 0].astype(int), labels.data.numpy()):
        writer.writerow([pid, survived])
print("Done!!!")
