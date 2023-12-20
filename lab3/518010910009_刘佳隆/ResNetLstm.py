from torch import nn

from lab3.LSTM import Lstm


class ResidualBlock(nn.Module):
    """
    定义ResNet的残差块。该残差块包含两个卷积层，每个卷积层后面跟着一个批归一化层。
    :param in_channels: 输入的通道数
    :param out_channels: 输出的通道数
    :param stride: 卷积的步长，默认为1
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(residual)
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    """
    定义ResNet模型，由多个残差块组成。
    :param latent_dim: 输出的特征维度
    """

    def __init__(self, latent_dim):
        super(ResNet, self).__init__()
        self.in_channels = 8
        self.conv1 = nn.Conv2d(3, 8, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_layer(8, 2)
        self.layer2 = self.make_layer(16, 2, stride=2)
        self.layer3 = self.make_layer(32, 2, stride=2)
        self.layer4 = self.make_layer(64, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, latent_dim)

    def make_layer(self, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        batch_size, time_steps, channel_x, height, width = x.shape
        conv_input = x.view(batch_size * time_steps, channel_x, height, width)
        x = self.conv1(conv_input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ResNetLstm(nn.Module):
    """
    使用自定义的ResNet和LSTM构建模型
    :param latent_dim: LSTM的输入维度
    :param hidden_size: LSTM的隐藏层维度
    :param lstm_layers: LSTM的层数
    :param bidirectional: LSTM是否为双向
    :param n_class: 分类的类别数
    """

    def __init__(self, latent_dim, hidden_size, lstm_layers, bidirectional, n_class):
        super(ResNetLstm, self).__init__()
        self.conv_model = ResNet(latent_dim=latent_dim)
        self.Lstm = Lstm(latent_dim, hidden_size, lstm_layers, bidirectional)
        self.output_layer = nn.Sequential(
            nn.Linear(2 * hidden_size if bidirectional == True else hidden_size, n_class),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        batch_size, time_steps, channel_x, height, width = x.shape
        conv_input = x.view(batch_size, time_steps, channel_x, height, width)
        conv_output = self.conv_model(conv_input)
        lstm_input = conv_output.view(batch_size, time_steps, -1)
        lstm_output = self.Lstm(lstm_input)
        lstm_output = lstm_output[:, -1, :]
        output = self.output_layer(lstm_output)
        return output
