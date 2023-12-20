# 总体介绍

该项目分别使用预训练的 resnet152 模型、自定义的 ConvNet 模型和自定义的 ResNet 模型，连接到 LSTM 模型上，实现了对视频的分类。

# 代码结构

- lab3_cnn_lstm.ipynb：训练代码，详细的代码解释和训练过程见其中的注释和 report.md
- lab3_data_scratch.py：数据预处理代码
- classify.py：使用训练好的模型对视频进行分类
- export_model.py：将训练好的模型导出为 `onnx` 格式
- LSTM.py, ConvLstm.py, PretrainedConvLstm.py, ResNetLstm.py：模型定义，主要用于 export_model.py 调用
- best_model_ConvLstm.pth：训练好的 ConvLstm 模型权重

# 训练方法

运行 `lab3-cnn-lstm.ipynb` 即可，三种模型均已在其中实现，通过修改训练时模型的名称即可选择不同的模型。

训练时须在第五步中修改模型路径、数据集路径、输出路径等参数。

# 使用方法

修改 `classify.py` 中的如下参数：

- `model_path` 为训练好的模型权重路径
- `model` 为训练时使用的模型，并修改对应的模型参数设置
- `video_path` 为待分类的视频路径

而后运行 `python classify.py` 即可。