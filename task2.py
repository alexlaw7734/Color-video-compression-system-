import matplotlib.pyplot as plt
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
    def __init__(self, p):
        super(Network, self).__init__()

        self.P = p

        self.module = nn.Sequential(
            nn.Linear(784, p),
            nn.Linear(p, 1568),
            nn.ReLU(),
            nn.Linear(1568, 784)
        )

    def forward(self, x):
        (b, _, _, _) = x.shape

        x = x.view(b, -1)
        x = self.module(x)
        return x.view(b, 1, 28, 28)


models, optimizers = [], []
for T in [10, 50, 200]:
    models.append(Network(T).to(device))
    optimizers.append(torch.optim.Adam(models[-1].parameters(), 1e-3))


# MARK: - train
mse = nn.MSELoss()

for model in models:
    model.train()

for epoch in range(10):
    losses = [[] for _ in range(3)]

    for batch, (xTrain, _) in enumerate(trainLoader):
        xTrain = xTrain.to(device)

        for i in range(3):
            optimizers[i].zero_grad()
            out = models[i](xTrain)
            loss = mse(out, xTrain)
            loss.backward()
            optimizers[i].step()

            losses[i].append(loss.item())

    print('Epoch: {}, P=10 Loss: {}, P=50 Loss: {}, P=200 Loss: {}'.format(epoch,
                                                                           np.average(losses[0]),
                                                                           np.average(losses[1]),
                                                                           np.average(losses[2])))

for model in models:
    torch.save(model, 'task2-{}.pt'.format(model.P))


# MARK: - 2a

for model in models:
    model.eval()

with torch.no_grad():
    for model in models:
        PSNRs = []
        for xTest, _ in testLoader:
            (b, _, _, _) = xTest.shape

            out = model(xTest.to(device)).cpu().numpy()

            max2 = np.amax(xTest.view(b, -1).numpy(), -1) ** 2

            loss = (xTest.numpy() - out) ** 2
            loss = np.reshape(loss, (b, -1))
            loss = np.average(loss, -1)

            PSNR = 10 * np.log10(max2 / loss)
            PSNRs = np.hstack((PSNRs, PSNR))

        print('P = {}, Avg PSNR: {:.2f}'.format(model.P, np.mean(PSNRs)))


# MARK: - 2b
def plotImages(f, tensors, baseIndex):
    arr = tensors.squeeze().cpu().numpy()
    for i in range(10):
        f.add_subplot(4, 10, baseIndex+i)
        plt.imshow(arr[i], cmap='gray')
        plt.axis('off')


fig = plt.figure(dpi=300)
customLoader = torch.utils.data.DataLoader(testDataset, batch_size=10, shuffle=True)
with torch.no_grad():
    xTen, _ = next(iter(customLoader))
    xTen = xTen.to(device)

    plotImages(fig, xTen, 1)
    for i in range(3):
        out = models[i](xTen)
        plotImages(fig, out, 11 + i * 10)

plt.savefig('task2.png')
