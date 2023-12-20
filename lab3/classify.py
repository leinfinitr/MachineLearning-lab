import albumentations
import cv2
import torch
from albumentations.pytorch import ToTensorV2

from lab3.ConvLstm import ConvLstm

# 使用第一次预训练的模型进行分类，模型参数如下：
num_classes = 10
batch_size = 4
num_workers = 4
device = 'cuda' if torch.cuda.is_available() else 'cpu'

num_frames = 20
img_size = (128, 128)
latent_dim = 2048
hid_size = 128
num_lstm_layers = 2
learning_rate = 2e-5

# 视频类别
classes = ['Biking', 'CliffDiving', 'Drumming', 'Haircut', 'HandstandWalking', 'HighJump', 'JumpingJack', 'Mixing',
           'Skiing', 'SumoWrestling']


def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()


def predict_video_category(video_path, model, transform):
    """
    对视频进行分类
    :param video_path: 视频路径
    :param model: 模型
    :param transform: 数据预处理
    :return:
    """
    frames = []
    vidcap = cv2.VideoCapture(video_path)

    while True:
        success, image = vidcap.read()
        if not success:
            break

        augmented = transform(image=image)
        image = augmented["image"]
        frames.append(image)

    frames = torch.stack(frames)
    frames = frames.unsqueeze(0)

    with torch.no_grad():
        output = model(frames)

    _, predicted_class = torch.max(output, dim=1)
    predicted_class = predicted_class.item()

    return classes[predicted_class]


# 数据预处理的转换流程。
# 使用albumentations库进行图像处理，包括图像大小调整、归一化和转换为张量。
transform = albumentations.Compose(
    [
        albumentations.Resize(height=img_size[0], width=img_size[1]),
        albumentations.Normalize(),
        ToTensorV2()
    ]
)

# 加载预训练的模型权重
model_path = "D:/course/MachineLearning/lab/lab3/result/best_model_ConvLstm.pth"
model = ConvLstm(latent_dim=latent_dim, hidden_size=hid_size, lstm_layers=num_lstm_layers,
                 bidirectional=True, n_class=num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))

# 对视频进行分类
video_path = "D:/course/MachineLearning/lab/lab3/data/HandstandWalking/v_HandstandWalking_g04_c01.avi"
predicted_class = predict_video_category(video_path, model, transform)
print("Predicted class:", predicted_class)
