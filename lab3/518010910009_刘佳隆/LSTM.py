from torch import nn


class Lstm(nn.Module):
    """
    定义LSTM模型
    :param latent_dim: LSTM的输入维度
    :param hidden_size: LSTM的隐藏层维度
    :param lstm_layers: LSTM的层数
    :param bidirectional: LSTM是否为双向
    """

    def __init__(self, latent_dim, hidden_size, lstm_layers, bidirectional):
        super(Lstm, self).__init__()
        self.Lstm = nn.LSTM(latent_dim, hidden_size, num_layers=lstm_layers, batch_first=True,
                            bidirectional=bidirectional)
        self.hidden_state = None

    # 重置LSTM的隐藏层状态
    def reset_hidden_state(self):
        self.hidden_state = None

    def forward(self, x):
        output, self.hidden_state = self.Lstm(x, self.hidden_state)
        return output
