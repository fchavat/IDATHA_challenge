from torch.nn import Module, Sequential, Conv2d, ReLU, MaxPool2d, Flatten, Linear, Softmax, Dropout

class CNN(Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = Sequential(
            Conv2d(3, 32, 3, 1),
            ReLU(),
            MaxPool2d(2), # out: 31x31x32
            Dropout(0.2),
            Conv2d(32, 64, 3, 1), # out: 29x29x64
            ReLU(),
            MaxPool2d(2), # out: 14x14x64
            Dropout(0.2),
            Conv2d(64, 128, 3, 1), # out: 12x12x128
            ReLU(),
            MaxPool2d(2), # out: 6x6x128
            Dropout(0.2),
            Conv2d(128, 128, 3, 2), # out: 2x2x128
            ReLU(),
            Flatten() # out: 1x1x512
        )

        self.classifier = Sequential(
            Linear(512, 128),
            Linear(128, num_classes),
            Softmax(dim=1)
        )

    def forward(self, img_batch):
        # in sape: BxCxHxW (batch, channels, height, width)
        res = self.cnn(img_batch)
        res = self.classifier(res)
        return res
