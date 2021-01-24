from sklearn.metrics import confusion_matrix
from utils import plotConfusionMatrix
import torch.nn as nn
import numpy as np
import torchvision
import torch

np.random.seed(1588390)
torch.manual_seed(1588390)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# MARK: - load data
trainDataset = torchvision.datasets.FashionMNIST('fashionMNIST',
                                                 train=True,
                                                 transform=torchvision.transforms.ToTensor(),
                                                 download=True)

testDataset = torchvision.datasets.FashionMNIST('fashionMNIST',
                                                train=False,
                                                transform=torchvision.transforms.ToTensor(),
                                                download=True)

trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=32, shuffle=True)
testLoader = torch.utils.data.DataLoader(testDataset, batch_size=32, shuffle=True)


# MARK: - define network
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU()
        )

        self.predict = nn.Sequential(
            nn.Linear(3136, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
            nn.LogSoftmax(-1)
        )

    def forward(self, x):
        (b, _, _, _) = x.shape
        out = self.backbone(x)
        out = out.view(b, -1)
        return self.predict(out)


model = Network().to(device)
optimizer = torch.optim.Adam(model.parameters(), 1e-3)

# MARK: - train
model.train()
for epoch in range(5):
    losses = []

    for batch, (xTrain, yTrain) in enumerate(trainLoader):
        xTrain, yTrain = xTrain.to(device), yTrain.to(device)

        optimizer.zero_grad()
        out = model(xTrain)
        loss = nn.functional.nll_loss(out, yTrain)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    print('Epoch: {}, Loss: {:.2f}'.format(epoch, np.average(losses)))

torch.save(model, 'task1.pt')


# MARK: - test
model.eval()
predictions, groundtruths = [], []

with torch.no_grad():
    for xTest, yTest in testLoader:
        out = model(xTest.to(device))
        yPred = out.argmax(-1).cpu().numpy().astype(np.uint8)

        predictions = np.hstack((predictions, yPred))
        groundtruths = np.hstack((groundtruths, yTest.numpy().astype(np.uint8)))

confusionMatrix = confusion_matrix(groundtruths, predictions)
plotConfusionMatrix(confusionMatrix, [i for i in range(10)], 'task1.png')
