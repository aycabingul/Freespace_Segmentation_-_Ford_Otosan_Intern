
import torch.nn as nn
import torch.nn.functional as F


class FoInternNet(nn.Module):
    def __init__(self, input_size, n_classes):
        super(FoInternNet, self).__init__()
        self.input_size = input_size
        self.n_classes = n_classes
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=n_classes, kernel_size=1, stride=1)

    def forward(self, x):

        x = self.conv1(x)

        x = F.relu(x)

        x = self.conv2(x)

        x = nn.Softmax(dim=1)(x)

        return x



if __name__ == '__main__':
    model = FoInternNet(input_size=(224, 224), n_classes=2)