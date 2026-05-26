from torch import nn
import torch.nn.init as init


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # 1x1 卷积
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = nn.BatchNorm2d(width)
        # 3x3 卷积
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)  # 归一化层
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        # 通过第一个 1x1 卷积层
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 通过第二个 3x3 卷积层
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # 通过第三个 1x1 卷积层
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # 跳跃连接：将输入（identity）与输出相加
        out += identity

        # 激活函数（ReLU）再次作用在输出上
        out = self.relu(out)

        return out


class BasicBlock(nn.Module):
    expansion = 1  # 输出通道不变

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, config, block, layers, num_classes=1000):
        super(ResNet, self).__init__()

        self.inplanes = config.encoder_channels[4]
        # 输入通道数为 in_channels，输出为 64，7x7 大卷积核，步长为 2，padding 为 3，不使用 bias,为一个CBR模块
        self.conv1 = nn.Conv2d(config.in_channels, config.encoder_channels[4], kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(config.encoder_channels[4])
        self.relu = nn.ReLU(inplace=True)

        # 最大池化层，3x3 核，步长为 2，无 padding，向上取整模式
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)  # change
        # 构建四个 stage 的残差网络
        # Stage1: 64 → 64 → 64 → 256
        self.layer1 = self._make_layer(block, config.encoder_channels[3] // 4, layers[0])
        # Stage2: 128 → 128 → 128 → 512
        self.layer2 = self._make_layer(block, config.encoder_channels[2] // 4, layers[1], stride=2)
        # Stage3: 256 → 256 → 256 → 1024
        self.layer3 = self._make_layer(block, config.encoder_channels[1] // 4, layers[2], stride=2)
        # Stage4: 512 → 512 → 512 → 2048
        self.layer4 = self._make_layer(block, config.encoder_channels[0] // 4, layers[3], stride=2)

        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(256 * block.expansion, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming初始化（即He初始化）
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                # BN层：γ=1, β=0
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                # 全连接层：小随机数初始化
                init.normal_(m.weight, 0, 0.01)
                init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        """
        构建一个残差层（stage），由多个残差块 block 组成。
            - block: 残差块类型（如 Bottleneck 或 BasicBlock）
            - planes: 当前 stage 的基础输出通道数
            - blocks: 当前 stage 包含的残差块数量
            - stride: 第一个 block 的步长（用于空间下采样）
            - block.expansion：通道扩展系数，resnet50=4
        """
        downsample = None
        # 在stage1-4的每第一个残差块，进行downsample
        # stage1：输入通道为64，第一个bottleneck需要进行downsample，参数为（K=1，s=1，p=0,c=4*64），图像尺寸不变，输出通道数变为64*4=256
        # stage2：输入通道为128，第一个bottleneck需要进行downsample，参数为（K=1，s=2，p=0,c=4*128），图像尺寸缩小为1/2，输出通道数变为128*4=512
        # stage3：输入通道为256，第一个bottleneck需要进行downsample，参数为（K=1，s=2，p=0,c=4*256），图像尺寸缩小为1/2，输出通道数变为256*4=1024
        # stage4：输入通道为512，第一个bottleneck需要进行downsample，参数为（K=1，s=2，p=0,c=4*512），图像尺寸缩小为1/2，输出通道数变为512*4=2048

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        feat1 = self.relu(x)

        x = self.maxpool(feat1)

        feat2 = self.layer1(x)  # stage 1 输出
        feat3 = self.layer2(feat2)  # stage 2 输出
        feat4 = self.layer3(feat3)  # stage 3 输出
        feat5 = self.layer4(feat4)  # stage 4 输出

        # 返回所有层的中间特征图（而非分类结果），常用于特征提取任务
        return [feat1, feat2, feat3, feat4, feat5]


def resnet34(**kwargs):
    """ResNet-34 模型"""
    # 使用 BasicBlock 结构，配置为 [3, 4, 6, 3]
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    del model.avgpool
    del model.fc
    return model


def resnet50(config, **kwargs):
    # 构建一个 ResNet-50 模型实例，使用 Bottleneck 结构
    # 每个 stage 分别包含 3, 4, 6, 3 个残差块（ResNet-50 配置）
    # 额外参数通过 kwargs 传入 ResNet 构造函数（如 num_classes, input_shape 等）
    model = ResNet(config=config, block=Bottleneck, layers=[3, 4, 6, 3], **kwargs)
    del model.avgpool
    del model.fc
    return model

def resnet50_light(config, **kwargs):
    # 构建一个 ResNet-50 模型实例，使用 Bottleneck 结构
    # 每个 stage 分别包含 3, 4, 6, 3 个残差块（ResNet-50 配置）
    # 额外参数通过 kwargs 传入 ResNet 构造函数（如 num_classes, input_shape 等）
    model = ResNet(config=config, block=Bottleneck, layers=[2, 2, 2, 2], **kwargs)
    del model.avgpool
    del model.fc
    return model


def resnet101(config, **kwargs):
    # ResNet-101 使用 Bottleneck 结构，stage 配置为 [3, 4, 23, 3]
    model = ResNet(config=config, block=Bottleneck, layers=[3, 4, 23, 3], **kwargs)
    del model.avgpool
    del model.fc
    # 返回去除分类头的 ResNet-101 模型
    return model
