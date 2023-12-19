# Lab Report

#### Author: *刘佳隆*

#### Student ID: *518010910009*

## 总体介绍

### 参考来源

- 参考代码1：https://www.kaggle.com/code/nguyenmanhcuongg/pytorch-video-classification-with-conv2d-lstm
- 参考代码2：https://github.com/doronharitan/human_activity_recognition_LRCN

### 代码结构

代码 lab3-cnn-lstm.ipynb 由以下几个部分组成：

#### 1. 定义数据集处理函数

参考 lab3_data_scratch.py 中的代码，定义数据集处理函数，用于将数据集转换为模型所需的数据格式。

#### 2. 定义模型

##### 调用预训练CNN搭建模型

使用预训练的ResNet152模型作为卷积层，搭建CNN-LSTM模型。

##### 使用自行定义的普通CNN搭建模型

自行搭建了一个由卷积层、池化层、全连接层和一个LSTM层组成的CNN-LSTM模型。

#### 3. 定义评估函数

定义了用于评估训练结果的函数，包括计算平均损失和准确率。定义了用于可视化训练过程中的损失和准确率的函数。

#### 4. 定义训练函数

参考代码 1 中的训练函数，对其进行了简化和修改，定义了训练函数，并在训练完成后保存模型的权重。

#### 5. 训练模型

指定模型参数，而后加载数据集和模型，调用训练函数进行训练。

#### 6. 评估模型

调用评估函数评估模型的训练结果，并生成可视化图表。

## 系统设计

### 模型设计

两种模型的 LSTM 结构均相同，使用 torch.nn.LSTM 搭建，其主要代码和函数解释如下：

``` python
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
```

#### 调用预训练CNN的模型设计

预训练的 CNN 使用 torchvision.models.resnet152 加载，固定卷积层的参数，修改最后一层全连接层，将其输出维度设置为 LSTM 的输入维度。

其主要代码和函数解释如下：

``` python
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
```

#### 自行定义的普通CNN的模型设计

自行定义的普通 CNN 使用 4 层 torch.nn.Conv2d、torch.nn.ReLU、torch.nn.MaxPool2d 和 1 层 torch.nn.Linear 搭建，将其输出维度设置为
LSTM 的输入维度。模型其余结构与调用预训练 CNN 的模型相同。

其主要代码和函数解释如下：

``` python
class Conv(nn.Module):
    """
    自定义的普通CNN模型
    :param latent_dim: 输出的特征维度
    """

    def __init__(self, latent_dim):
        super(Conv, self).__init__()
        self.conv_model = nn.Sequential(
            # 输入维度：(batch_size, 3, 128, 128)
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=6, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 输出维度：(batch_size, 64, 32, 32)

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 输出维度：(batch_size, 64, 16, 16)

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=2), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 输出维度：(batch_size, 64, 8, 8)
            
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 输出维度：(batch_size, 64, 4, 4)
        )
        self.fc = nn.Linear(64 * 4 * 4, latent_dim)

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
```

### 训练方法

训练函数主要包括：

1. 训练前先对模型进行评估，而后设定缺省的训练参数，而后开始训练。
2. 训练时先通过模型计算输出，而后计算损失，而后反向传播计算梯度，而后更新参数。
3. 每轮训练完成后，对模型进行评估，并根据学习率调度器调整学习率。
4. 训练完成后保存模型的权重。

主要代码解释如下：

``` python
def train(model, train_data, loss_fn, optimizer, epochs, device, save_last_weights_path=None,
          save_best_weights_path=None, steps_per_epoch=None,
          validation_data=None, scheduler=None):
    """
    训练模型
    :param model: 要训练的模型。
    :param train_data: 训练数据集的数据加载器。
    :param loss_fn: 损失函数。
    :param optimizer: 优化器。
    :param epochs: 训练的轮数。
    :param device: 训练设备。
    :param save_last_weights_path: 可选参数，保存最后模型权重的路径。
    :param save_best_weights_path: 可选参数，保存最佳模型权重的路径。
    :param steps_per_epoch: 可选参数，每个epoch的步数。
    :param validation_data: 可选参数，用于验证的数据加载器。
    :param scheduler: 可选参数，学习率调度器。
    :return: 
    """

    if save_best_weights_path:
        # 评估当前模型在验证数据集上的损失
        best_loss, _ = evaluate(model, validation_data, loss_fn, device)

    if steps_per_epoch is None:
        # 如果没有指定每个epoch的步数，则将其设置为训练数据集的长度
        steps_per_epoch = len(train_data)

    num_steps = len(train_data)
    iterator = iter(train_data)
    count_steps = 1

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_acc': [],
        'val_loss': []
    }

    # 将模型移动到设备上
    model = model.to(device)

    # 遍历每个epoch
    for epoch in range(1, epochs + 1):

        running_loss = 0.
        train_correct = 0
        train_total = steps_per_epoch * train_data.batch_size

        model.train()

        for step in tqdm(range(steps_per_epoch), desc=f'epoch: {epoch}/{epochs}: ', ncols=100):
            img_batch, label_batch = next(iterator)
            img_batch, label_batch = img_batch.to(device), label_batch.to(device)
            # 将梯度置零
            optimizer.zero_grad()
            # 前向传播计算输出
            output_batch = model(img_batch)
            # 计算损失
            loss = loss_fn(output_batch, label_batch.long())
            # 反向传播计算梯度
            loss.backward(retain_graph=True)
            # 更新参数
            optimizer.step()
            # 预测标签
            _, predicted_labels = torch.max(output_batch.data, dim=1)
            # 统计正确预测的数量
            train_correct += (label_batch == predicted_labels).sum().item()
            # 计算平均损失
            running_loss += loss.item()
            # 打印训练损失和准确率
            if count_steps == num_steps:
                # 循环迭代器，以便继续训练数据集的下一个epoch
                count_steps = 0
                iterator = iter(train_data)
            count_steps += 1

        train_loss = running_loss / steps_per_epoch
        train_accuracy = train_correct / train_total

        if scheduler:
            # 如果提供了学习率调度器，则根据训练损失调整学习率
            scheduler.step(train_loss)

        history['train_loss'].append(float(train_loss))
        history['train_acc'].append(float(train_accuracy))

        # 评估模型在验证数据集上的性能
        val_loss, val_acc = evaluate(model, validation_data, loss_fn, device)
        # 打印训练损失和准确率
        print(
            f'epoch: {epoch}, train_accuracy: {train_accuracy:.2f}, loss: {train_loss:.3f}, val_accuracy: {val_acc:.2f}, val_loss: {val_loss:.3f}')

        if save_best_weights_path:
            if val_loss < best_loss:
                # 如果验证损失更小，则保存模型的权重
                best_loss = val_loss
                torch.save(model.state_dict(), save_best_weights_path)
                print(f'Saved successfully best weights to:', save_best_weights_path)
        history['val_loss'].append(float(val_loss))
        history['val_acc'].append(float(val_acc))

    if save_last_weights_path:
        # 如果提供了保存最后权重的路径，则保存模型的权重
        torch.save(model.state_dict(), save_last_weights_path)
        print(f'Saved successfully last weights to:', save_last_weights_path)

    return model, history
```

## 训练过程 & 调参实验及结果

### 主要参数说明

在训练过程中不变的参数设置和说明如下：

| 参数名         | 参数值  | 说明         |
|-------------|------|------------|
| num_classes | 10   | 视频分类的类别数   |
| batch_size  | 4    | 每个batch的大小 |
| num_workers | 4    | 数据加载器的线程数  |
| device      | cuda | 训练设备       |

与训练过程相关的参数说明如下：

| 参数名             | 说明         |
|-----------------|------------|
| num_frames      | 每个视频的帧数    |
| img_size        | 图像的像素大小    |
| latent_dim      | CNN的输出维度   |
| hid_size        | LSTM的隐藏层维度 |
| num_lstm_layers | LSTM的层数    |
| learning_rate   | 学习率        |

### 调用预训练CNN的模型训练过程及结果

#### 第一次训练

参数设置如下：

| 参数名             | 参数值        |
|-----------------|------------|
| num_frames      | 15         |
| img_size        | (128, 128) |
| latent_dim      | 2048       |
| hid_size        | 128        |
| num_lstm_layers | 2          |
| learning_rate   | 2e-5       |

训练过程如下：

![](auto_coding-master/model/distilgpt2_fine_tuned_coder_1/train_loss.png)

评估结果如下：

![](auto_coding-master/model/distilgpt2_fine_tuned_coder_1/dev_eval_scores.png)

#### 第二次训练

参数设置如下：

| 参数名             | 参数值      |
|-----------------|----------|
| num_frames      | 20       |
| img_size        | (64, 64) |
| latent_dim      | 2048     |
| hid_size        | 128      |
| num_lstm_layers | 2        |
| learning_rate   | 2e-5     |

训练过程如下：

![](auto_coding-master/model/distilgpt2_fine_tuned_coder_2/train_loss.png)

评估结果如下：

![](auto_coding-master/model/distilgpt2_fine_tuned_coder_2/dev_eval_scores.png)

#### 第三次训练

参数设置如下：

| 参数名             | 参数值      |
|-----------------|----------|
| num_frames      | 20       |
| img_size        | (64, 64) |
| latent_dim      | 1024     |
| hid_size        | 64       |
| num_lstm_layers | 2        |
| learning_rate   | 2e-5     |

训练过程如下：

![](auto_coding-master/model/distilgpt2_fine_tuned_coder_3/train_loss.png)

评估结果如下：

![](auto_coding-master/model/distilgpt2_fine_tuned_coder_3/dev_eval_scores.png)

#### 对比

### 调用自行定义的普通CNN的模型训练过程及结果

