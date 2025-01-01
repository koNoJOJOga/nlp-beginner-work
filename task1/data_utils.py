import numpy as np

def load_data(file_path, max_samples=None, label_map=None):
    """加载数据集并返回标签和句子列表"""
    labels, sentences = [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if max_samples and idx >= max_samples:
                break
            label, sentence = line.strip().split('\t', 1)
            if label_map:
                label = label_map[label]  # 使用映射将标签转换为数字
            labels.append(int(label))  # 转换为整数标签
            sentences.append(sentence)
    return np.array(labels), sentences  # 返回数字类型标签

def vectorize_text(sentences, ngram_range=(1, 1), max_features=5000):
    """将文本转化为特征向量"""
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(ngram_range=ngram_range, max_features=max_features)
    X = vectorizer.fit_transform(sentences)
    return X.toarray(), vectorizer
