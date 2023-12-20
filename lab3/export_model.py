import torch

from lab3.ConvLstm import ConvLstm
from lab3.ResNetLstm import ResNetLstm

# 固定的参数
num_classes = 10
batch_size = 4
num_workers = 4
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 可调整的参数
num_frames = 15  # You can adjust this to balance speed and accuracy
img_size = (128, 128)  # You can adjust this to balance speed and accuracy
latent_dim = 2048
hid_size = 128
num_lstm_layers = 2
learning_rate = 2e-5

ConvLstmModel = ConvLstm(latent_dim=latent_dim, hidden_size=hid_size, lstm_layers=num_lstm_layers, bidirectional=True,
                         n_class=num_classes)
torch.onnx.export(ConvLstmModel, torch.randn((1, 15, 3, 128, 128)), 'ConvLstm.onnx')

ResNetLstmModel = ResNetLstm(latent_dim=latent_dim, hidden_size=hid_size, lstm_layers=num_lstm_layers,
                             bidirectional=True,
                             n_class=num_classes)
torch.onnx.export(ResNetLstmModel, torch.randn((1, 15, 3, 128, 128)), 'ResNetLstm.onnx')
