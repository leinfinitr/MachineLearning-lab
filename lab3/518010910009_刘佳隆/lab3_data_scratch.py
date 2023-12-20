import os

import albumentations as A
import cv2 as cv
import torch
from albumentations.pytorch import ToTensorV2
from loguru import logger
from sklearn.model_selection import train_test_split


class VideoDataset(torch.utils.data.Dataset):
    """
    Custom Dataset for loading videos and their class labels
    """

    def __init__(self, data_dir, num_classes=10, num_frames=20, transform=None, target_transform=None):
        super().__init__()

        self.data_dir = data_dir

        self.transform = transform
        self.target_transform = target_transform
        self.num_classes = num_classes
        self.num_frames = num_frames

        self.video_filename_list = []
        self.classesIdx_list = []

        self.class_dict = {class_label: idx for idx, class_label in enumerate(
            sorted(os.listdir(self.data_dir)))}

        for class_label, class_idx in self.class_dict.items():
            class_dir = os.path.join(self.data_dir, class_label)
            for video_filename in sorted(os.listdir(class_dir)):
                self.video_filename_list.append(
                    os.path.join(class_label, video_filename))
                self.classesIdx_list.append(class_idx)

    # 返回数据集中视频的数量
    def __len__(self):
        return len(self.video_filename_list)

    # 读取视频文件，并进行帧的采样和数据预处理。返回采样后的帧序列。
    def read_video(self, video_path):
        frames = []
        cap = cv.VideoCapture(video_path)
        count_frames = 0
        while True:
            ret, frame = cap.read()
            if ret:
                if self.transform:
                    transformed = self.transform(image=frame)
                    frame = transformed['image']

                frames.append(frame)
                count_frames += 1
            else:
                break

        stride = count_frames // self.num_frames
        new_frames = []
        count = 0
        for i in range(0, count_frames, stride):
            if count >= self.num_frames:
                break
            new_frames.append(frames[i])
            count += 1

        cap.release()

        return torch.stack(new_frames, dim=0)

    # 返回数据集中索引idx对应的视频及其类别标签
    def __getitem__(self, idx):
        classIdx = self.classesIdx_list[idx]
        video_filename = self.video_filename_list[idx]
        video_path = os.path.join(self.data_dir, video_filename)
        frames = self.read_video(video_path)
        return frames, classIdx


num_classes = 10
batch_size = 4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_frames = 20  # You can adjust this to balance speed and accuracy
img_size = (128, 128)  # You can adjust this to balance speed and accuracy
num_workers = 4

# 数据预处理的转换流程。
# 使用albumentations库进行图像处理，包括图像大小调整、归一化和转换为张量。
transform = A.Compose(
    [
        A.Resize(height=img_size[0], width=img_size[1]),
        A.Normalize(),
        ToTensorV2()
    ]
)

if __name__ == '__main__':

    logger.info('Loading dataset')
    # 加载数据集并指定数据集的路径、帧数、类别数和数据预处理的转换函数。
    full_dataset = VideoDataset(data_dir="data", num_frames=num_frames, num_classes=num_classes, transform=transform)
    # 将数据集分为训练集和测试集，其中测试集的比例为0.2。
    train_dataset, test_dataset = train_test_split(full_dataset, test_size=0.2, random_state=42)
    # 使用PyTorch的DataLoader加载数据集。
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                              num_workers=num_workers)
    logger.info('Dataset loaded')

    for batch_idx, (data, target) in enumerate(train_loader):
        print(data.shape)
        print(target.shape)
        break
