from torch import nn

from lab3.LSTM import Lstm


class Conv(nn.Module):
    """
    自定义的普通CNN模型
    :param latent_dim: 输出的特征维度
    """

    def __init__(self, latent_dim):
        super(Conv, self).__init__()
        self.conv_model = nn.Sequential(
            # 输入维度：(batch_size, 3, 128, 128)
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=6, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 输出维度：(batch_size, 16, 32, 32)

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 输出维度：(batch_size, 32, 16, 16)

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 输出维度：(batch_size, 64, 4, 4)

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 输出维度：(batch_size, 64, 2, 2)
        )
        self.fc = nn.Linear(64 * 2 * 2, latent_dim)

    def forward(self, x):
        batch_size, time_steps, channel_x, height, width = x.shape
        x = x.view(batch_size * time_steps, channel_x, height, width)
        x = self.conv_model(x)
        x = x.view(batch_size * time_steps, -1)
        x = self.fc(x)
        x = x.view(batch_size, time_steps, -1)
        return x


class ConvLstm(nn.Module):
    """
    使用自定义的CNN和LSTM构建模型
    :param latent_dim: LSTM的输入维度
    :param hidden_size: LSTM的隐藏层维度
    :param lstm_layers: LSTM的层数
    :param bidirectional: LSTM是否为双向
    :param n_class: 分类的类别数
    """

    def __init__(self, latent_dim, hidden_size, lstm_layers, bidirectional, n_class):
        super(ConvLstm, self).__init__()
        self.conv_model = Conv(latent_dim)
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
