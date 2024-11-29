import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.neighbors import BallTree

# Tải dữ liệu MNIST
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist.data, mnist.target.astype(int)

# Xây dựng Ball Tree
ball_tree = BallTree(X, leaf_size=40)

# Lưu mô hình
import joblib
joblib.dump((ball_tree, y), "ball_tree_mnist.pkl")
