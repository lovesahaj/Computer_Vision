from PIL import Image
import torch
from torchvision import transforms
import torchvision
import torch.nn as nn
from torchvision.datasets import EMNIST
from utils import load_checkpoint, save_checkpoint
import cv2
import matplotlib.pyplot as plt
import xgboost as xgb

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# DEVICE = 'cpu'
LEARNING_RATE = 1e-4
BATCH_SIZE = 128
IMG_SIZE = 64
IMG_CHANNELS = 1
Z_DIM = 100
NUM_EPOCH = 50
FEATURES_DISC = 64
FEATURES_GEN = 64
NUM_WORKERS = 4
LOAD_MODEL = False


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 11)

        self.max_pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.relu = nn.ReLU()
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.dropout1(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = self.log_softmax(x)
        return output


def train(model, device, train_loader, optimizer, epoch, criterion):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        total_loss = 0

        # if torch.randn(1).item() > 0:
        #     data_, target_ = (torch.zeros_like(data).to(
        #         device) * -0.4242), (torch.ones_like(target) * 10).to(device)

        #     optimizer.zero_grad()
        #     output_ = model(data_)
        #     loss_ = criterion(output_, target_)
        #     loss_.backward()
        #     optimizer.step()
        #     total_loss += loss_

        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), total_loss.item()))


def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    empty_len = 0
    with torch.no_grad():
        for data, target in test_loader:
            # if torch.randn(1).item() > 0:
            #     data_, target_ = (torch.zeros_like(data).to(
            #         device) * -0.4242), (torch.ones_like(target) * 10).to(device)

            #     output_ = model(data_)
            #     # sum up batch loss
            #     test_loss += criterion(output_, target_).sum().item()
            #     # get the index of the max log-probability
            #     pred_ = output_.argmax(dim=1, keepdim=True)
            #     correct += pred_.eq(target_.view_as(pred_)).sum().item()
            #     empty_len += len(data_)

            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += criterion(output, target).sum().item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, (len(test_loader.dataset) + empty_len),
        100. * correct / (len(test_loader.dataset) + empty_len)))


def main():
    device = DEVICE
    train_transform = transforms.Compose([
        transforms.RandomRotation(45),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.RandomAutocontrast(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_tranforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset1 = EMNIST('../edata', train=True, download=True,
                      transform=train_transform, split='digits')
    dataset2 = EMNIST('../edata', train=False, download=True,
                      transform=test_tranforms, split='digits')

    train_loader = torch.utils.data.DataLoader(dataset1, batch_size=BATCH_SIZE,
                                               shuffle=True, num_workers=NUM_WORKERS)

    test_loader = torch.utils.data.DataLoader(dataset2, batch_size=BATCH_SIZE,
                                              shuffle=False, num_workers=NUM_WORKERS)

    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    if LOAD_MODEL:
        load_checkpoint("emnist.pth.tar", model)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=1, gamma=0.1)

    criterion = nn.NLLLoss()
    for epoch in range(1, NUM_EPOCH + 1):
        train(model, device, train_loader, optimizer, epoch, criterion)
        test(model, device, test_loader, criterion)
        scheduler.step()

        checkpoint = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }

        save_checkpoint(checkpoint, "emnist.pth.tar")


def predict(image, model):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image = (image - image.mean()) / (image.max() - image.min())

    image = Image.fromarray(image)

    tranform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    image = tranform(image)
    # plt.imshow(image[0])
    # plt.show()

    model.eval()
    with torch.no_grad():
        image = image.unsqueeze(0).to(device)
        pred = (model(image))
        # print(pred)
        pred = torch.argmax(pred, dim=1)
        # print(pred)

    pred = pred.item()
    if pred == 10:
        return 0

    return pred


def predict_xg(image):
    model = xgb.Booster()
    model.load_model('xg_model.json')

    image = (image - image.mean()) / image.std()

    image = Image.fromarray(image)

    tranform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    image = tranform(image)

    image = image.flatten(start_dim=1).numpy()

    image = xgb.DMatrix(image)

    return int(model.predict(image)[0])


def give_arr(image):
    model = Net().to('cuda' if torch.cuda.is_available() else 'cpu')
    load_checkpoint(torch.load("emnist.pth.tar"), model)

    step = 100
    arr = list()
    for i in range(9):
        temp_arr = list()
        for j in range(9):
            temp_image = image[i * step: (i + 1) * step,
                               j * step: (j + 1) * step]

            pred = predict(temp_image, model)
            temp_arr.append(pred)
        arr.append(temp_arr)

    return arr


def give_arr_xg(image):
    step = 100
    arr = list()
    for i in range(9):
        temp_arr = list()
        for j in range(9):
            temp_image = image[i * step: (i + 1) * step,
                               j * step: (j + 1) * step]

            pred = predict_xg(temp_image)
            temp_arr.append(pred)
        arr.append(temp_arr)

    return arr


def show_image(image):
    while True:
        cv2.imshow("Image", image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


def xgboost_train(trainLoader, testLoader):
    param_list = [("eta", 0.08),
                  ("max_depth", 6),
                  ("subsample", 0.8),
                  ("colsample_bytree", 0.8),
                  ("objective", "multi:softmax"),
                  ("eval_metric", "merror"),
                  ("alpha", 8),
                  ("lambda", 2),
                  ("num_class", 10),
                  ('tree_method', 'gpu_hist'),
                  ('gpu_id', 0)
                  ]
    n_rounds = 600
    early_stopping = 50

    X_train, y_train = trainLoader.dataset.data.numpy(
    ).reshape((-1, 784)), trainLoader.dataset.targets.numpy().reshape((-1))
    X_valid, y_valid = testLoader.dataset.data.numpy().reshape(
        (-1, 784)), testLoader.dataset.targets.numpy().reshape((-1))

    d_train = xgb.DMatrix(X_train, label=y_train)
    d_val = xgb.DMatrix(X_valid, label=y_valid)
    eval_list = [(d_train, "train"), (d_val, "validation")]
    bst = xgb.train(param_list, d_train, n_rounds, evals=eval_list,
                    early_stopping_rounds=early_stopping, verbose_eval=True)

    bst.save_model('xg_model.json')


if __name__ == "__main__":
    main()
