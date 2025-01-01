import numpy as np

class LogisticRegression:
    def __init__(self, input_dim, output_dim, learning_rate=0.01):
        self.W = np.zeros((input_dim, output_dim))
        self.b = np.zeros(output_dim)
        self.learning_rate = learning_rate

    def fit(self, X, y, epochs=10, batch_size=32):
        # 转换输入为 numpy 数组
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(y, np.ndarray):
            y = np.array(y)

        for epoch in range(epochs):
            # 随机打乱数据
            indices = np.random.permutation(len(X))  # 生成随机整数数组
            X, y = X[indices], y[indices]

            for i in range(0, len(X), batch_size):
                X_batch = X[i:i + batch_size]
                y_batch = y[i:i + batch_size]
                logits = X_batch @ self.W + self.b
                probs = self.softmax(logits)
                gradients_W = X_batch.T @ (probs - self.one_hot(y_batch, self.b.shape[0])) / batch_size
                gradients_b = np.mean(probs - self.one_hot(y_batch, self.b.shape[0]), axis=0)
                self.W -= self.learning_rate * gradients_W
                self.b -= self.learning_rate * gradients_b

    def softmax(self, logits):
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    def one_hot(self, y, num_classes):
        one_hot_labels = np.zeros((len(y), num_classes))
        one_hot_labels[np.arange(len(y)), y] = 1
        return one_hot_labels

    def predict(self, X):
        logits = X @ self.W + self.b
        return np.argmax(logits, axis=1)
