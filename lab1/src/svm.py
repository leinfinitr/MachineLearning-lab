# 通过 SVM 完成二分类任务
import time

import matplotlib.pyplot as plt
import numpy as np

# 读取数据
X_test = np.loadtxt(open("../lab1_dataset/X_test.csv"), delimiter=",", skiprows=1)
X_train = np.loadtxt(open("../lab1_dataset/X_train.csv"), delimiter=",", skiprows=1)
Y_test = np.loadtxt(open("../lab1_dataset/Y_test.csv"), delimiter=",", skiprows=1)
Y_train = np.loadtxt(open("../lab1_dataset/Y_train.csv"), delimiter=",", skiprows=1)

# 数据处理
# 将 Y_train 和 Y_test 中的 0 替换为 -1
for i in range(Y_train.shape[0]):
    if Y_train[i] == 0:
        Y_train[i] = -1
for i in range(Y_test.shape[0]):
    if Y_test[i] == 0:
        Y_test[i] = -1


# 函数定义
def HingeLoss(x, y, w, b, lamda):
    # 计算平均损失
    return np.mean(np.maximum(0, 1 - y * (np.dot(x, w) + b))) + lamda * np.dot(w.T, w) / 2


def grad(x, y, w, b, lamda):
    dot = (np.dot(x, w) + b) * y
    gradient = np.zeros((x.shape[1]))
    gradient_b = 0
    for i in range(dot.shape[0]):
        if dot[i] < 1:
            gradient += -y[i] * x[i]
            gradient_b += -y[i]
    gradient = gradient.T
    gradient += lamda * w * x.shape[0]
    gradient /= x.shape[0]
    gradient_b /= x.shape[0]
    return gradient, gradient_b


def evaluate(x, y, w, b, s):
    y_pred = np.dot(x, w) + b
    for j, i in enumerate(y_pred):
        if i < 0:
            y_pred[j] = -1
        else:
            y_pred[j] = 1

    accuracy = np.mean(y_pred == y)
    print(s, accuracy)
    return accuracy


def gradientdescent(x, y):
    w = np.zeros((x.shape[1]))
    b = np.zeros(1)
    iteration = 100000  # number of iteration
    learning_reta = 0.00005  # learning rate
    loss_v = []  # loss value
    evaluate_train_v = []  # train accuracy
    evaluate_test_v = []  # test accuracy
    i = 0  # iteration time
    while True:
        gradient, gradient_b = grad(x, y, w, b, 0.1)
        w = w - (learning_reta * gradient)
        b = b - (learning_reta * gradient_b)

        # 记录 loss
        l = HingeLoss(x, y, w, b, 0.1)
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
plt.title('Support Vector Machine')
plt.xlabel('iteration')
plt.ylabel('accuracy')
plt.legend()
plt.show()
