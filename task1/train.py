import torch
from torch.nn import CrossEntropyLoss
import torch.optim as optim
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os

class Trainer:
    def __init__(self, model, device, learning_rate=0.01, patience=5, save_path="./best_model.pth"):
        self.model = model
        self.device = device
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=3, factor=0.5, verbose=True)
        self.patience = patience
        self.save_path = save_path
        self.best_val_loss = float('inf')
        self.early_stop_counter = 0
        self.model.to(device)

    def train(self, train_loader, val_loader, num_epochs):
        """模型训练过程"""
        for epoch in range(1, num_epochs + 1):
            print(f"Epoch {epoch}/{num_epochs}")
            train_loss = self._train_one_epoch(train_loader)
            val_loss, val_accuracy = self.evaluate(val_loader)

            # 调整学习率
            self.scheduler.step(val_loss)

            # Early Stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.early_stop_counter = 0
                self.save_model(self.save_path)  # 保存当前最佳模型
                print("Model improved. Saved!")
            else:
                self.early_stop_counter += 1
                print(f"Early stop counter: {self.early_stop_counter}/{self.patience}")
                if self.early_stop_counter >= self.patience:
                    print("Early stopping triggered.")
                    break

    def _train_one_epoch(self, train_loader):
        """单个epoch的训练过程"""
        self.model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc="Training"):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            self.optimizer.zero_grad()

            # 注意：MPS 不支持 torch.cuda.amp.autocast()
            outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            total_loss += loss.item()
            loss.backward()
            self.optimizer.step()

        avg_loss = total_loss / len(train_loader)
        print(f"Training loss: {avg_loss}")
        return avg_loss

    def evaluate(self, val_loader):
        """模型评估过程"""
        self.model.eval()
        total_loss = 0
        correct, total = 0, 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()

                predictions = torch.argmax(outputs.logits, dim=-1)
                correct += (predictions == torch.argmax(labels, dim=1)).sum().item()
                total += labels.size(0)

        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        print(f"Validation loss: {avg_loss}, Validation accuracy: {accuracy}")
        return avg_loss, accuracy

    def save_model(self, file_path):
        """保存模型权重"""
        torch.save(self.model.state_dict(), file_path)

    def load_model(self, file_path):
        """加载模型权重"""
        if os.path.exists(file_path):
            self.model.load_state_dict(torch.load(file_path))
            self.model.to(self.device)
            print(f"Model loaded from {file_path}")
        else:
            print(f"File {file_path} does not exist.")
