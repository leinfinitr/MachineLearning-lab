# 通过 MLP 完成二分类任务
import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn

# 读取数据
X_test = np.loadtxt(open("../lab1_dataset/X_test.csv"), delimiter=",", skiprows=1)
X_train = np.loadtxt(open("../lab1_dataset/X_train.csv"), delimiter=",", skiprows=1)
Y_test = np.loadtxt(open("../lab1_dataset/Y_test.csv"), delimiter=",", skiprows=1)
Y_train = np.loadtxt(open("../lab1_dataset/Y_train.csv"), delimiter=",", skiprows=1)


# 函数定义
def evaluate(x, y, s):
    x = torch.from_numpy(x).float()
    prediction = neural_network(x)
    result = np.zeros(y.shape[0])
    for i in range(prediction.shape[0]):
        if prediction[i][0] > prediction[i][1]:
            result[i] = 0
        else:
            result[i] = 1

    accuracy = np.mean(result == y)
    print(s, accuracy)
    return accuracy


# 构建模型
n_hidden_1 = 20  # 第一层隐藏层的神经元个数
n_hidden_2 = 20  # 第二层隐藏层的神经元个数
n_input = 29  # 输入层的神经元个数
n_classes = 2  # 输出层的神经元个数

neural_network = nn.Sequential(
    nn.Linear(n_input, n_hidden_1),
    nn.ReLU(),
    nn.Linear(n_hidden_1, n_hidden_2),
    nn.ReLU(),
    nn.Linear(n_hidden_2, n_classes),
    nn.Sigmoid()
)

# 优化器
optimizer = torch.optim.SGD(neural_network.parameters(), lr=0.01)
# 损失函数
loss_func = nn.CrossEntropyLoss()

# 训练
epoch = 100000  # 训练次数
loss_v = []  # loss value
start_time = time.time()
evaluate_train_v = []  # train accuracy
evaluate_test_v = []  # test accuracy
for i in range(epoch):
    # 将数据转换为 Tensor
    x = torch.from_numpy(X_train).float()
    y = torch.from_numpy(Y_train).long()

    # 前向传播
    prediction = neural_network(x)

    # 计算损失
    loss = loss_func(prediction, y)

    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 记录 loss
    loss_v.append(loss.item())

    # 每迭代 1000 次，输出一次 loss、记录对准确度的预测
    if i % 1000 == 0:
        print('iteration: ', i, ' loss: ', loss.item())
        evaluate_test_v.append(evaluate(X_test, Y_test, 'test accuracy: '))
        evaluate_train_v.append(evaluate(X_train, Y_train, 'train accuracy: '))

end_time = time.time()
print('time: ', end_time - start_time)

plt.plot(range(len(loss_v)), loss_v)
plt.xlabel('iteration')
plt.ylabel('loss')
plt.show()

plt.plot(range(len(evaluate_train_v)), evaluate_train_v, label='train')
plt.plot(range(len(evaluate_test_v)), evaluate_test_v, label='test')
plt.xlabel('iteration')
plt.ylabel('accuracy')
plt.legend()
plt.show()
