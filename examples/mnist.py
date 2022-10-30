import argparse
import random

import numpy as np
from PIL import Image
from torchvision.datasets import MNIST

import tinynn
import tinynn.nn as nn
import tinynn.nn.functional as F
from tinynn.utils.data import DataLoader


class MnistModel(nn.Module):
    def __init__(self, num_classes=10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        # self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.dropout1 = nn.Dropout(p=0.25)
        self.fc1 = nn.Linear(in_features=32 * 13 * 13, out_features=128)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x: tinynn.Tensor) -> tinynn.Tensor:
        x = self.conv1(x)
        x = F.relu(x)
        # x = self.conv2(x)
        # x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = self.dropout1(x)
        x = x.flatten(start_dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        logits = self.fc2(x)
        return logits


class PILToTensor:
    def __call__(self, pic: Image.Image) -> tinynn.Tensor:
        x = tinynn.tensor(np.asarray(pic) / 255)
        if x.ndim == 2:
            x = x[None, :, :]
        assert x.ndim == 3
        return x


class Normalize:
    def __init__(self, mean, std) -> None:
        self.mean = mean
        self.std = std

    def __call__(self, x: tinynn.Tensor) -> tinynn.Tensor:
        return (x - self.mean) / self.std


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--max_epoch", type=int, default=8)
    parser.add_argument("--train_batch_size", type=int, default=100)
    parser.add_argument("--val_batch_size", type=int, default=1000)
    parser.add_argument("--log_interval", type=int, default=50)
    args = parser.parse_args()

    print(f"Train config: {vars(args)}")

    random.seed(args.seed)
    np.random.seed(args.seed)

    input_transform = Compose([PILToTensor(), Normalize(mean=0.1307, std=0.3081)])
    train_data = MNIST(
        root="../data", train=True, transform=input_transform, download=True
    )
    val_data = MNIST(root="../data", train=False, transform=input_transform)

    train_loader = DataLoader(
        train_data, batch_size=args.train_batch_size, shuffle=True
    )
    val_loader = DataLoader(val_data, batch_size=args.val_batch_size, shuffle=False)

    model = MnistModel()
    optimizer = tinynn.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    for epoch_idx in range(args.max_epoch):
        # train one epoch
        for batch_idx, (inputs, target) in enumerate(train_loader):
            logits = model(inputs)

            loss = F.cross_entropy(logits, target)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            acc = logits.argmax(dim=1).eq(target).mean()

            if (batch_idx + 1) % args.log_interval == 0:
                print(
                    f"[TRAIN] epoch: {epoch_idx}/{args.max_epoch}, batch: {batch_idx}/{len(train_loader)}, "
                    f"loss: {loss.item():.4f}, acc: {acc.item():.4f}"
                )

        # evaluation
        val_acc = 0
        val_loss = 0
        for batch_idx, (inputs, target) in enumerate(val_loader):
            with tinynn.no_grad():
                logits = model(inputs)
            val_loss += F.cross_entropy(logits, target, reduction="sum").item()
            val_acc += logits.argmax(dim=1).eq(target).sum().item()
        val_acc /= len(val_data)
        val_loss /= len(val_data)

        print(
            f"\n[VALIDATE] epoch: {epoch_idx}/{args.max_epoch}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}\n"
        )


if __name__ == "__main__":
    main()
