import torch
from torch import nn
from torch.autograd import Function
from torch.nn import functional as F

class GradReverseLayer(Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()

class Discriminator(nn.Module):
    def __init__(self, input_dim=128):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
                nn.Linear(input_dim, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, 2)
                )

    def forward(self, x):
        x = GradReverseLayer.apply(x)
        out = self.discriminator(x)
        return out

class ConvLarge(nn.Module):
    def __init__(self, input_dim=3, num_classes=10, stochastic=True, top_bn=False):
        super(ConvLarge, self).__init__()
        self.block1 = self.conv_block(input_dim, 128, 3, 1, 1, 0.1)
        self.block2 = self.conv_block(128, 128, 3, 1, 1, 0.1)
        self.block3 = self.conv_block(128, 128, 3, 1, 1, 0.1)

        self.block4 = self.conv_block(128, 256, 3, 1, 1, 0.1)
        self.block5 = self.conv_block(256, 256, 3, 1, 1, 0.1)
        self.block6 = self.conv_block(256, 256, 3, 1, 1, 0.1)

        self.block7 = self.conv_block(256, 512, 3, 1, 0, 0.1)
        self.block8 = self.conv_block(512, 256, 1, 1, 0, 0.1)
        self.block9 = self.conv_block(256, 128, 1, 1, 0, 0.1)

        self.AveragePooling = nn.AdaptiveAvgPool2d((1, 1))
        
        maxpool = [nn.MaxPool2d(kernel_size=2, stride=2)]
        if stochastic: maxpool.append(nn.Dropout2d())
        self.maxpool = nn.Sequential(*maxpool)
        
        classifier = [nn.Linear(128, num_classes)]
        if top_bn: classifier.append(nn.BatchNorm1d(num_classes))
        self.classifier = nn.Sequential(*classifier)

    def conv_block(self, input_dim, out_dim, kernel_size=3, stride=1, padding=1, lrelu_slope=0.01):
        return nn.Sequential(
                nn.Conv2d(input_dim, out_dim, kernel_size, stride, padding, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.LeakyReLU(inplace=True, negative_slope=lrelu_slope)
                )

    def forward(self, x, return_feature=False):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.maxpool(out)

        out = self.block4(out)
        out = self.block5(out)
        out = self.block6(out)
        out = self.maxpool(out)

        out = self.block7(out)
        out = self.block8(out)
        out = self.block9(out)

        feature = self.AveragePooling(out)
        feature = feature.view(feature.shape[0], -1)
        logits = self.classifier(feature)
        
        if return_feature:
            return logits, feature
        else:
            return logits
    
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
    model = ConvLarge(input_dim=3)
    discriminator = Discriminator(input_dim=128)

    img = torch.randn(100, 3, 32, 32)
    logits, feature = model(img, return_feature=True)
    flag = discriminator(feature)
    print(feature.shape)
    print(logits.shape)
    print(flag.shape)

