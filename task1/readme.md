# 任务一：基于机器学习的文本分类

实现基于 Logistic Regression 的文本分类

## 参考

   1. [文本分类](文本分类.md)
   2. 《[神经网络与深度学习](https://nndl.github.io/)》 第2/3章

## 数据

数据集采用 CNews 数据集，数据包括 10 个类别：体育、财经、房产、家居、教育、科技、时尚、时政、游戏、娱乐。每个类别有 6500 条数据，训练集：5000 条/类别，验证集：500 条/类别，测试集：1000 条/类别。

数据集的下载链接：[百度网盘](https://pan.baidu.com/s/10ZI4Y-0eecSHdSnPeJEnMg?pwd=3y4n) 提取码：3y4n。下载后解压，将文件放入 `datasets` 目录中：

- `cnews.train.txt`：训练集（50000条）
- `cnews.val.txt`：验证集（5000条）
- `cnews.test.txt`：测试集（10000条）

## 运行

1. **训练模型**：
    使用以下命令训练模型：

    ```bash
    python main.py --config config.yml
    ```

2. **测试模型（单条语句）**：
    使用以下命令进行单条语句测试：

    ```bash
    python test.py --config config.yml
    ```

## 步骤概览

1. **加载数据集**：
   - 从 `datasets` 目录加载 `cnews.train.txt`，`cnews.val.txt` 和 `cnews.test.txt` 文件。
   - 使用 `data_utils.py` 中的 `load_data` 函数加载并处理数据。
   - 使用标签映射将类别标签转换为整数类型。

2. **数据预处理**：
   - 使用 `CountVectorizer` 对文本数据进行向量化，支持 N-gram 特征表示。
   - 使用 `vectorize_text` 函数将文本数据转化为稀疏矩阵形式的特征向量。

3. **模型训练**：
   - 使用 `LogisticRegression` 模型进行训练，支持批量梯度下降优化。
   - 调用 `fit` 方法来训练模型，并在每个 epoch 计算模型参数的梯度。

4. **模型评估**：
   - 在验证集上评估模型的准确率。
   - 使用训练好的模型在测试集上进行预测并评估测试集准确率。

5. **保存和加载模型**：
   - 模型训练完成后，将模型参数和向量化器保存到磁盘。
   - 使用 `pickle` 保存模型及其参数，便于后续加载和预测。

## 文件结构

```
├── config.yml            # 配置文件，包含训练参数、文件路径等
├── datasets              # 数据集文件夹，包含 cnews.train.txt 等数据文件
├── main.py               # 主程序，负责加载数据、训练模型并评估
├── model.py              # 定义 Logistic Regression 模型
├── test.py               # 用于测试训练好的模型
├── data_utils.py         # 数据加载与预处理
└── saved_models          # 存放训练后的模型和向量器
```

## 代码解释

### **main.py**

- 负责加载训练、验证和测试数据集。
- 向量化文本数据，并将其输入到 Logistic Regression 模型进行训练。
- 评估模型在验证集和测试集上的准确性，并保存模型和向量器。

### **test.py**

- 负责加载已保存的模型和向量器，并在测试集上进行预测和评估。

### **model.py**

- 定义 `LogisticRegression` 类，包含模型的训练、预测及优化算法。
- 支持使用批量梯度下降（Batch Gradient Descent）进行训练。

### **data_utils.py**

- 负责加载数据、向量化文本以及处理文本特征。
- 提供 `load_data` 函数来加载文本数据，并将标签映射为数字类型。

## 备注

- 本项目使用 NumPy 实现了一个简单的 Logistic Regression 分类器。
- 模型的训练和评估过程都可以通过修改 `config.yml` 文件中的参数来进行定制，例如学习率、批次大小、N-gram 范围等。