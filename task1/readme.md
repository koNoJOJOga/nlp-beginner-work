# 任务一：基于机器学习的文本分类

实现基于robeta的的文本分类

## 参考

   1. [文本分类](文本分类.md)
   2. 《[神经网络与深度学习](https://nndl.github.io/)》 第2/3章
 
## 数据

数据集采用gaussic的数据集，https://github.com/gaussic/text-classification-cnn-rnn<br />
链接: https://pan.baidu.com/s/10ZI4Y-0eecSHdSnPeJEnMg?pwd=3y4n 提取码: 3y4n 
下载文件中的cnews.train.txt，放到datasets目录下即可<br />

## 运行

训练：python main.py<br />
单条语句测试：python test.py <br />

## 步骤概览

1. 加载 RoBERTa 模型和 tokenizer
2. 准备数据集
3. 定义模型
4. 训练和评估模型