### 代码结构

- `analysis.py`：分析结果并绘制相关图表
- `convert.py`：将原始数据集转换为 json 格式
- `data.py`：根据选定的模型和语言，调用 `model.tokenizer` 将转化后的数据加载到模型中
- `evaluate.py`：评估模型的性能
- `interact.py`：加载训练好的模型并启动交互式代码生成
- `model.py`：定义模型的结构
- `my_train.ipynb`：在 Google Colab 或 Kaggle 上训练模型的 notebook
- `trainer.py`：定义模型的训练过程
- `train.py`：加载数据集并设置模型框架、训练参数、文件路径等，然后开始训练

### 使用方法

#### 使用数据集训练

受限于笔记本配置，本地训练较慢，因此迁移至 Google Colab 和 Kaggle进行训练。将转化后的数据集和初始模型上传至 Google
Drive 或 Kaggle Datasets，而后修改 `my_train.ipynb` 中的文件路径，运行 `my_train.ipynb` 即可。

本地训练只需要修改 `train.py` 中的文件路径，然后运行 `train.py` 即可。

#### 运行模型

将训练结果解压至 `./model` 目录下，然后修改 `interact.py` 中 `'--model_path'` 参数为 `"/model/[$model_name]"`
，然后运行 `interact.py` 即可。