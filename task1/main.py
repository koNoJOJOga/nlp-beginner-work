import torch
from transformers import RobertaTokenizer
from model import *
from train import *
from datasets.dataloader import *

# 数据集文件路径
train_file = './datasets/cnews.train.txt'
val_file = './datasets/cnews.val.txt'
test_file = './datasets/cnews.test.txt'

# 类别
categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']

# 超参数
num_epochs = 3
device = torch.device("mps" if torch.cuda.is_available() else "cpu")

# 创建数据加载器
data_loader_creator = DataLoaderCreator()
train_loader, val_loader, test_loader = data_loader_creator.prepare_data(
    train_file, val_file, test_file, categories
)

# 加载模型
roberta_model = RobertaModel(num_labels=len(categories)).get_model()

# 初始化训练器
trainer = Trainer(model=roberta_model, device=device)

# 训练与评估
trainer.train(train_loader=train_loader, val_loader=val_loader, num_epochs=num_epochs)

# 保存模型
trainer.save_model('./model/roberta_finetuned.pth')
