import pickle
import numpy as np
from data_utils import load_data

def test_model(config):
    # 加载测试数据
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

    test_labels, test_sentences = load_data(config['test_data_path'], label_map=label_map)

    # 加载模型和向量器
    with open(config['model_path'], 'rb') as f:
        model_data = pickle.load(f)
    model_W, model_b = model_data

    with open(config['vectorizer_path'], 'rb') as f:
        vectorizer = pickle.load(f)

    # 向量化测试数据
    X_test = vectorizer.transform(test_sentences).toarray()

    # 初始化模型
    from model import LogisticRegression
    input_dim = X_test.shape[1]
    output_dim = len(label_map)
    model = LogisticRegression(input_dim=input_dim, output_dim=output_dim)
    model.W = model_W
    model.b = model_b

    # 测试模型
    y_test_pred = model.predict(X_test)
    test_accuracy = (y_test_pred == test_labels).mean()
    print(f"Test Accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":
    import yaml
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    test_model(config)
