# 通过逻辑回归完成二分类任务
import time

import matplotlib.pyplot as plt
import numpy as np

# 读取数据
X_test = np.loadtxt(open("../lab1_dataset/X_test.csv"), delimiter=",", skiprows=1)
X_train = np.loadtxt(open("../lab1_dataset/X_train.csv"), delimiter=",", skiprows=1)
Y_test = np.loadtxt(open("../lab1_dataset/Y_test.csv"), delimiter=",", skiprows=1)
Y_train = np.loadtxt(open("../lab1_dataset/Y_train.csv"), delimiter=",", skiprows=1)


# 函数定义
def sigmoid(x, w, b):
    dot = np.dot(x, w) + b
    # 对于 dot 中的元素，根据正负性分别进行处理
    for i in range(dot.shape[0]):
        if dot[i] > 0:
            dot[i] = 1 / (1 + np.exp(-dot[i]))
        else:
            dot[i] = np.exp(dot[i]) / (1 + np.exp(dot[i]))
    return dot


def loss(r, y):
    # 计算平均损失
    # 此处加上 pow(10, -6) 是为了防止 log(0) 的出现
    return np.mean(-(r * np.log(y + pow(10, -6))) - ((1 - r) * np.log(1 - y + pow(10, -6))))


def grad(x, y, r):
    # 此处除以 x.shape[0] 是为了防止梯度过大
    return -np.dot(x.T, (r - y)) / x.shape[0]


def evaluate(x, y, w, b, s):
    y_pred = sigmoid(x, w, b)
    for j, i in enumerate(y_pred):
        if i < 0.5:
            y_pred[j] = 0
        else:
            y_pred[j] = 1

    accuracy = np.mean(y_pred == y)
    print(s, accuracy)
    return accuracy


def gradientdescent(x, r):
    w = np.zeros((x.shape[1]))
    b = np.zeros(1)
    iteration = 100000  # number of iteration
    learning_reta = 0.0001  # learning rate
    loss_v = []  # loss value
    evaluate_train_v = []  # train accuracy
    evaluate_test_v = []  # test accuracy
    i = 0  # iteration time
    while True:
        y = sigmoid(x, w, b)
        gradient = grad(x, y, r)
        w = w - (learning_reta * gradient)
        b = b - (learning_reta * np.sum(y - r) / x.shape[0])

        # 记录 loss
        l = loss(r, y)
        loss_v.append(l)

        # 每迭代 1000 次，输出一次 loss、记录对准确度的预测
        if i % 1000 == 0:
            print('iteration: ', i, ' loss: ', l)
            evaluate_test_v.append(evaluate(X_test, Y_test, w, b, 'test accuracy: '))
            evaluate_train_v.append(evaluate(X_train, Y_train, w, b, 'train accuracy: '))

        # 终止条件
        # 1. 迭代次数达到上限
        if i == iteration:
            break
        # 2. loss 收敛
        #    为了防止 loss 的震荡，l 与 loss_v[i - 10] 进行比较
        #    如果 loss 的变化小于 0.00001，则认为 loss 收敛
        # if i > 10 and abs(l - loss_v[i - 10]) < 0.00001:
        #     break

        i += 1
    return w, b, loss_v, evaluate_train_v, evaluate_test_v


start_time = time.time()
w, b, loss_v, evaluate_train_v, evaluate_test_v = gradientdescent(X_train, Y_train)
end_time = time.time()
print('time: ', end_time - start_time)

plt.plot(range(len(loss_v)), loss_v)
plt.xlabel('iteration')
plt.ylabel('loss')
plt.show()

plt.plot(range(len(evaluate_train_v)), evaluate_train_v, label='train')
plt.plot(range(len(evaluate_test_v)), evaluate_test_v, label='test')
plt.title('Logistic Regression')
plt.xlabel('iteration')
plt.ylabel('accuracy')
plt.legend()
plt.show()
