import pickle
import numpy as np
from data_utils import load_data, vectorize_text
from model import LogisticRegression

def main(config):
    # 定义标签映射
    label_map = {
        '体育': 0,
        '财经': 1,
        '房产': 2,
        '家居': 3,
        '教育': 4,
        '科技': 5,
        '时尚': 6,
        '时政': 7,
        '游戏': 8,
        '娱乐': 9,
    }

    # 加载数据
    print("Loading datasets...")
    train_labels, train_sentences = load_data(config['train_data_path'], label_map=label_map)
    val_labels, val_sentences = load_data(config['val_data_path'], label_map=label_map)
    test_labels, test_sentences = load_data(config['test_data_path'], label_map=label_map)

    # 向量化数据
    print("Vectorizing text...")
    X_train, vectorizer = vectorize_text(train_sentences, ngram_range=(1, config['ngram']), max_features=config['max_features'])
    X_val = vectorizer.transform(val_sentences).toarray()
    X_test = vectorizer.transform(test_sentences).toarray()

    # 初始化模型
    print("Initializing model...")
    input_dim = X_train.shape[1]
    output_dim = len(label_map)
    model = LogisticRegression(input_dim=input_dim, output_dim=output_dim, learning_rate=config['learning_rate'])

    # 训练模型
    print("Training model...")
    model.fit(X_train, train_labels, epochs=config['epochs'], batch_size=config['batch_size'])

    # 验证模型
    print("Validating model...")
    y_val_pred = model.predict(X_val)
    accuracy = (y_val_pred == val_labels).mean()
    print(f"Validation Accuracy: {accuracy:.4f}")

    # 测试模型
    print("Testing model...")
    y_test_pred = model.predict(X_test)
    test_accuracy = (y_test_pred == test_labels).mean()
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # 保存模型和向量器
    print("Saving model and vectorizer...")
    with open(config['model_path'], 'wb') as f:
        pickle.dump((model.W, model.b), f)
    with open(config['vectorizer_path'], 'wb') as f:
        pickle.dump(vectorizer, f)

if __name__ == "__main__":
    import yaml
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    main(config)
