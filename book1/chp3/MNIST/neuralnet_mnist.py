import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.datasets import mnist

# Sigmoid 함수 정의
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Softmax 함수 정의
def softmax(x):
    if x.ndim == 2:
        x = x - np.max(x, axis=1, keepdims=True)
        x = np.exp(x)
        return x / np.sum(x, axis=1, keepdims=True)
    x = x - np.max(x)  # 오버플로우 방지
    return np.exp(x) / np.sum(np.exp(x))

def get_data():
    # tensorflow를 이용해서 MNIST 데이터셋을 로드하고, 데이터 정규화(0~1 사이로)
    (x_train, t_train), (x_test, t_test) = mnist.load_data()
    x_test = x_test.reshape(-1, 28*28)  # 이미지를 1차원 배열로 평탄화
    x_test = x_test.astype('float32') / 255.0  # 데이터를 0~1 범위로 정규화
    return x_test, t_test

def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network

def predict(network, x):
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, w1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, w3) + b3
    y = softmax(a3)

    return y

x, t = get_data()
network = init_network()

batch_size = 100  # 배치 크기
accuracy_cnt = 0

for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

print(f"Accuracy: {float(accuracy_cnt) / len(x):.2f}")
