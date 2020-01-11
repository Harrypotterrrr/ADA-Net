import torch
from torch import nn
from torch.autograd import Function
from torch.nn import functional as F

class identical(nn.Module):

    def forward(self, input):
        return input

class GradReverseLayer(Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()

class Classifier(nn.Module):
    def __init__(self, input_dim=128, num_class=10, top_bn=False):
        super().__init__()

        self.top_bn = top_bn
        self.fc = nn.Linear(input_dim, num_class)
        self.top_bn_layer = nn.BatchNorm1d(num_class)

    def forward(self, x):
        out = self.fc(x)
        if self.top_bn:
            out = self.top_bn_layer(out)
        out = F.softmax(out, 1) # regularization TODO
        return out

class Discriminator(nn.Module):
    def __init__(self, input_dim=128, lambd=0.0):
        super().__init__()

        self.lambd = lambd
        self.discriminator = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = GradReverseLayer.apply(x)
        out = self.discriminator(x)
        return out

class ResNet(nn.Module):
    def __init__(self, input_dim=3, stochastic=True, lambd=0.0):
        super().__init__()
        self.block1 = self.conv_block(input_dim, 128, 3, 1, padding=1, lrelu_slope=0.1)
        self.block2 = self.conv_block(128, 128, 3, 1, 1, 0.1)
        self.block3 = self.conv_block(128, 128, 3, 1, 1, 0.1)

        self.block4 = self.conv_block(128, 256, 3, 1, 1, 0.1)
        self.block5 = self.conv_block(256, 256, 3, 1, 1, 0.1)
        self.block6 = self.conv_block(256, 256, 3, 1, 1, 0.1)

        self.block7 = self.conv_block(256, 512, 3, 1, 1, 0.1)
        self.block8 = self.conv_block(512, 256, 3, 1, 1, 0.1)
        self.block9 = self.conv_block(256, 128, 3, 1, 1, 0.1)

        self.AveragePooling = nn.AdaptiveAvgPool2d((1, 1))

        self.max_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1), # padding conf TODO
            nn.Dropout2d() if stochastic else identical()
        )


    def conv_block(self, input_dim, out_dim, kernel_size=3, stride=1, padding=1, lrelu_slope=0.01):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=input_dim,
                out_channels=out_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU(inplace=True, negative_slope=lrelu_slope) # inplace conf TODO
        )

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.max_pool(out)

        out = self.block4(out)
        out = self.block5(out)
        out = self.block6(out)
        out = self.max_pool(out)

        out = self.block7(out)
        out = self.block8(out)
        out = self.block9(out)
        feature = self.AveragePooling(out)
        feature = feature.view(feature.shape[0], -1)

        return feature


class SimpleNet(nn.Module):

    def __init__(self, in_channel, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(in_features=in_channel, out_features=10)
        self.fc2 = nn.Linear(10, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out, inplace=True)
        return self.fc2(out)


if __name__ == '__main__':

    resnet = ResNet(input_dim=3)
    classifier = Classifier(num_class=10)
    disc = Discriminator(input_dim=128)

    img = torch.randn(100, 3, 32, 32)
    feature = resnet(img)
    label = classifier(feature)
    flag = disc(feature)
    print(feature.shape)
    print(label.shape)
    print(flag.shape)
