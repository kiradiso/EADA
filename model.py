import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np


class ResidualBlock(nn.Module):
    """Residual Block."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim_out, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim_out, affine=True))
        # pytorch __setattr__ will check it^s type is parameter, buffer, module and normal object
        # it use orderdict to manage it, parameter is the subclass of variable and it will automatically
        # add to parameter dict when cal __setattr__ in module.

    def forward(self, x):
        return x + self.main(x)



class Generator_f(nn.Module):
    """Use features as input, Generator, Feature extractor"""
    def __init__(self, feature_size, rep_num=0, div=2, out_size=256, use_bn=True, drop_out=False, p=0.3):
        assert not (use_bn and drop_out)
        super(Generator_f, self).__init__()

        cur_size = feature_size
        layers = []
        for i in range(rep_num):
            layers.append(nn.Linear(cur_size, cur_size//div))
            cur_size = cur_size // div
            if use_bn:
                layers.append(nn.BatchNorm1d(cur_size, affine=True))
            elif drop_out:
                layers.append(nn.Dropout(p=p))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(cur_size, out_size))
        if use_bn:
            layers.append(nn.BatchNorm1d(out_size, affine=True))
        elif drop_out:
            layers.append(nn.Dropout(p=p))
        self.main = nn.Sequential(*layers)
        print(self)

    def forward(self, x):
        return self.main(x)


class Generator_Alx(nn.Module):
    """fine-tune AlexNet"""
    def __init__(self, feature_size=256, fine_tune=False):
        super(Generator_Alx, self).__init__()
        model = models.alexnet(pretrained=True)
        self.features = model.features
        mod = list(model.classifier.children())
        mod.pop()
        self.linear = nn.Sequential(*mod)
        if not fine_tune:
            for param in self.features.parameters():
                param.requires_grad = False
            for param in self.linear.parameters():
                param.requires_grad = False
        self.extra = torch.nn.Linear(4096, feature_size)
        print(self)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.linear(x)
        return self.extra(x)


class Generator(nn.Module):
    """Generator. Feature extractor."""
    def __init__(self, image_size, conv_dim=64, ds_num=2, res_num=2, feature_size=512):
        super(Generator, self).__init__()

        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.BatchNorm2d(conv_dim, affine=True))
        layers.append(nn.ReLU(inplace=True))
        self.cur_imsize = image_size

        # Down-Sampling
        curr_dim = conv_dim
        for i in range(ds_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(curr_dim*2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2
            self.cur_imsize //= 2
        for i in range(res_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # To Linear
        self.linear = nn.Sequential(
            nn.Linear(curr_dim*pow(self.cur_imsize, 2), 1024),
            nn.ReLU(inplace=True), # inplace: may be the input and the output are same
            nn.Linear(1024, feature_size), # here not add the relu miaomiaomiao
            nn.LeakyReLU(inplace=True)
        )

        self.main = nn.Sequential(*layers)
        print(self)

    def forward(self, x):
        # replicate spatially and concatenate domain information
        mid_out = self.main(x)
        mid_out = mid_out.view(mid_out.size()[0], -1)
        return self.linear(mid_out)


class Classifier(nn.Module):
    """Classifier and both C, Gd, Gyd can use this !! this classifier accept features"""
    def __init__(self, feature_size=512, repeat_num=0, class_num=2, to_prob=False, use_bn=True, drop_out=False, p=0.3):
        assert not (use_bn and drop_out)
        super(Classifier, self).__init__()

        self.curdim = feature_size
        self.outnum = class_num if class_num > 2 else 1
        layers = []
        for i in range(repeat_num):
            layers.append(nn.Linear(self.curdim, self.curdim//2))
            self.curdim //= 2
            if use_bn:
                layers.append(nn.BatchNorm1d(self.curdim, affine=True))
            elif drop_out:
                layers.append(nn.Dropout(p=p))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(self.curdim, self.outnum))
        if to_prob:
            if self.outnum == 1:
                layers.append(nn.Sigmoid())
            else:
                layers.append(nn.Softmax())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class Classifier_withsize(nn.Module):
    """Classifier and both C, Gd, Gyd can use this !! this classifier accept features"""
    def __init__(self, feature_size=512, size=[1024, 1024], class_num=2, to_prob=False, use_bn=True, drop_out=False, p=0.3):
        assert not (use_bn and drop_out)
        super(Classifier_withsize, self).__init__()

        cur_size = feature_size
        self.outnum = class_num if class_num > 2 else 1
        layers = []
        for i in range(len(size)):
            layers.append(nn.Linear(cur_size, size[i]))
            cur_size = size[i]
            if use_bn:
                layers.append(nn.BatchNorm1d(cur_size, affine=True))
            elif drop_out:
                layers.append(nn.Dropout(p=p))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(cur_size, self.outnum))
        if to_prob:
            if self.outnum == 1:
                layers.append(nn.Sigmoid())
            else:
                layers.append(nn.Softmax())
        self.main = nn.Sequential(*layers)
        print(self)

    def forward(self, x):
        return self.main(x)


class Decoder(nn.Module):
    """Decoder accept features of Gf to reconstruct the image"""
    def __init__(self, feature_size=512, ini_dim=256, ini_imsize=64, us_num=2):
        super(Decoder, self).__init__()
        change_size = ini_dim * pow(ini_imsize, 2)
        self.linear = nn.Sequential(
            nn.Linear(feature_size, change_size),
            nn.BatchNorm1d(change_size),
            nn.ReLU()
        )
        self.ini_dim = ini_dim
        self.ini_imsize = ini_imsize
        self.us_num = us_num
        layers = []
        for i in range(us_num):
            layers.append(nn.ConvTranspose2d(ini_dim, ini_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(ini_dim//2))
            layers.append((nn.ReLU(inplace=True)))
            ini_dim = ini_dim//2

        layers.append(nn.Conv2d(ini_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        mid_out = self.linear(x)
        mid_out = mid_out.view(mid_out.size()[0], self.ini_dim, self.ini_imsize, self.ini_imsize)
        return self.main(mid_out)


if __name__ == "__main__":
    a = Generator_Alx()