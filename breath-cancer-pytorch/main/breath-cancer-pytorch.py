import numpy as np
from sklearn.metrics import accuracy_score, f1_score,confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

path = '/media/duytrieu/Work/UbuntuWork/PycharmProjects/breath-cancer-pytorch/corpus/breast-cancer-wisconsin.data'
with open(path, 'r') as f:
    lines = f.readlines()
# lines = lines[:2]
lines=[l.strip().replace('?','5').split(',') for l in lines]
lines = np.array(lines)
ids = lines[:,0].astype(np.int)
x = lines[:,1:-1].astype(np.float)

y = lines[:,-1].astype(np.int)
y = y/2-1

scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)

epoch = 12000
h = 100
lr = 0.01

class MyModule(nn.Module):

    def __init__(self):
        super(MyModule, self).__init__()
        self.layer1  = nn.Linear(9,h)
        self.sigmoid = nn.LogSigmoid()
        self.layer2  = nn.Linear(h,h)
        self.layer3  = nn.Linear(h,2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.sigmoid(x)
        x = self.layer2(x)
        x = self.sigmoid(x)
        x = self.layer3(x)
        return x

model = MyModule()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
weight = torch.FloatTensor(np.array([1,2]))
criterion = nn.CrossEntropyLoss(weight=weight)

features = Variable(torch.from_numpy(x).type(torch.FloatTensor))
target = Variable(torch.from_numpy(y).type(torch.LongTensor))

log = []
for e in range(epoch):
    y_hat = model(features)
    loss = criterion(y_hat, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if e % 200 == 0:
        _, y_pred = torch.max(y_hat, 1)
        y_pred = y_pred.data.numpy()

        acc = accuracy_score(y, y_pred)
        p, r, f, _ = precision_recall_fscore_support(y, y_pred)
        # cm = confusion_matrix(y, y_pred)
        log.append((e, loss.data[0], f[0], f[1]))
    if e % 1000 == 0:
        print('Epoch %d: %f' % (e, loss))
print('DONE')

from matplotlib import pyplot as plt
epochs, losses, f0, f1 = zip(*log)
figure = plt.plot(epochs, losses, 'r-',epochs,f0,'b-',epochs,f1,'g-')
plt.show()