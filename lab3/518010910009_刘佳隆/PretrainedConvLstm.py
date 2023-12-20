import torchvision
from torch import nn

from lab3.LSTM import Lstm


class PretrainedConv(nn.Module):
    """
    使用预训练的ResNet152模型作为卷积层
    :param latent_dim: 输出的特征维度
    """

    def __init__(self, latent_dim):
        super(PretrainedConv, self).__init__()
        # 使用预训练的ResNet152模型
        self.conv_model = torchvision.models.resnet152(pretrained=True)
        # ====== 固定卷积层的参数 ======
        for param in self.conv_model.parameters():
            param.requires_grad = False
        # ====== 修改最后一层全连接层 ======
        # latent_dim为输出的特征维度，也是LSTM的输入维度
        self.conv_model.fc = nn.Linear(self.conv_model.fc.in_features, latent_dim)

    def forward(self, x):
        return self.conv_model(x)


class PretrainedConvLstm(nn.Module):
    """
    使用预训练的CNN和LSTM构建模型
    :param latent_dim: LSTM的输入维度
    :param hidden_size: LSTM的隐藏层维度
    :param lstm_layers: LSTM的层数
    :param bidirectional: LSTM是否为双向
    :param n_class: 分类的类别数
    """

    def __init__(self, latent_dim, hidden_size, lstm_layers, bidirectional, n_class):
        super(PretrainedConvLstm, self).__init__()
        self.conv_model = PretrainedConv(latent_dim)
        self.Lstm = Lstm(latent_dim, hidden_size, lstm_layers, bidirectional)
        self.output_layer = nn.Sequential(
            nn.Linear(2 * hidden_size if bidirectional == True else hidden_size, n_class),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        batch_size, time_steps, channel_x, height, width = x.shape
        conv_input = x.view(batch_size * time_steps, channel_x, height, width)
        conv_output = self.conv_model(conv_input)
        lstm_input = conv_output.view(batch_size, time_steps, -1)
        lstm_output = self.Lstm(lstm_input)
        lstm_output = lstm_output[:, -1, :]
        output = self.output_layer(lstm_output)
        return output
