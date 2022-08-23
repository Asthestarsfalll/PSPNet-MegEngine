import megengine as mge
import megengine.module as M
import megengine.functional as F
from megengine import hub
from .resnet import *
from .pooling import AdaptiveAvgPooling2D


class Dropout2d(M.Module):
    def __init__(self, p):
        super(Dropout2d, self).__init__()
        assert 0 <= p and p <= 1
        self.p = p

    def forward(self, x):
        if not self.training and self.p == 0:
            return x
        assert x.ndim == 4
        c = x.shape[1]
        keep_prob = mge.tensor(1 - self.p, dtype=x.dtype)
        random_tensor = mge.random.normal(mean=0, std=1, size=(c, ))
        random_tensor = F.floor(random_tensor)
        random_tensor = random_tensor.reshape(1, -1, 1, 1)
        return x * random_tensor


class PPM(M.Module):
    def __init__(self, in_dim, reduction_dim, bins):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(M.Sequential(
                AdaptiveAvgPooling2D(bin),
                M.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                M.BatchNorm2d(reduction_dim),
                M.ReLU()
            ))
        self.features = self.features

    def forward(self, x):
        x_size = x.shape
        out = [x]
        for f in self.features:
            out.append(F.nn.interpolate(
                f(x), x_size[2:], mode='bilinear', align_corners=True))
        return F.concat(out, axis=1)


class PSPNet(M.Module):
    def __init__(
            self,
            layers=50,
            bins=(1, 2, 3, 6),
            dropout=0.1,
            classes=2,
            zoom_factor=8,
            use_ppm=True,
            criterion=None
        ):
        super(PSPNet, self).__init__()
        assert layers in [50, 101, 152]
        assert 2048 % len(bins) == 0
        assert classes > 1
        assert zoom_factor in [1, 2, 4, 8]
        self.zoom_factor = zoom_factor
        self.use_ppm = use_ppm
        self.criterion = criterion

        # TODO: add pretrained model
        if layers == 50:
            resnet = resnet50()
        elif layers == 101:
            resnet = resnet101()
        else:
            resnet = resnet152()
        self.layer0 = M.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.conv2,
            resnet.bn2,
            resnet.relu,
            resnet.conv3,
            resnet.bn3,
            resnet.relu,
            resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

        fea_dim = 2048
        if use_ppm:
            self.ppm = PPM(fea_dim, int(fea_dim/len(bins)), bins)
            fea_dim *= 2
        self.cls = M.Sequential(
            M.Conv2d(fea_dim, 512, kernel_size=3, padding=1, bias=False),
            M.BatchNorm2d(512),
            M.ReLU(),
            Dropout2d(p=dropout),
            M.Conv2d(512, classes, kernel_size=1)
        )
        self.aux = M.Sequential(
            M.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
            M.BatchNorm2d(256),
            M.ReLU(),
            Dropout2d(p=dropout),
            M.Conv2d(256, classes, kernel_size=1)
        )

    def forward(self, x, y=None):
        x_size = x.shape
        assert (x_size[2]-1) % 8 == 0 and (x_size[3]-1) % 8 == 0
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x_tmp = self.layer3(x)
        x = self.layer4(x_tmp)
        if self.use_ppm:
            x = self.ppm(x)
        x = self.cls(x)
        if self.zoom_factor != 1:
            x = F.nn.interpolate(x, size=(h, w), mode='bilinear',
                                     align_corners=True)

        if self.training:
            aux = self.aux(x_tmp)
            if self.zoom_factor != 1:
                aux = F.nn.interpolate(aux, size=(
                    h, w), mode='bilinear', align_corners=True)
            main_loss = self.criterion(x, y)
            aux_loss = self.criterion(aux, y)
            return x.max(1)[1], main_loss, aux_loss
        else:
            return x

@hub.pretrained(
    "https://studio.brainpp.com/api/v1/activities/3/missions/75/files/b3ee9ded-07bb-4e97-9688-82bce4160f99"
)
def pspnet50(**kwargs):
    return PSPNet(layers=50, **kwargs)


def pspnet101(**kwargs):
    return PSPNet(layers=101, **kwargs)


def pspnet152(**kwargs):
    return PSPNet(layers=152, **kwargs)
