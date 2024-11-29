import os
import numpy as np
from flask import Flask, request, render_template, url_for
from sklearn.neighbors import BallTree
from sklearn.datasets import fetch_openml
from PIL import Image

app = Flask(__name__) #http://127.0.0.1:5000/

# Đường dẫn để lưu ảnh tải lên
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Tải và xử lý dữ liệu MNIST
def load_mnist_data():
    if not os.path.exists("train_data.npy") or not os.path.exists("train_labels.npy"):
        print("Đang tải dữ liệu MNIST...")
        mnist = fetch_openml('mnist_784', version=1)
        data = mnist.data / 255.0  # Chuẩn hóa từ 0-255 thành 0-1
        labels = mnist.target.astype(int)  # Chuyển nhãn thành số nguyên
        np.save("train_data.npy", data)
        np.save("train_labels.npy", labels)
        print("Dữ liệu MNIST đã được lưu.")
    else:
        print("Tải dữ liệu MNIST từ tệp đã lưu...")
    train_data = np.load("train_data.npy")
    train_labels = np.load("train_labels.npy")
    return train_data, train_labels

train_data, train_labels = load_mnist_data()

# Tạo mô hình BallTree
tree = BallTree(train_data)

# Hàm dự đoán chữ số từ ảnh
def predict_digit(filepath):
    img = Image.open(filepath).convert('L')  # Chuyển ảnh sang grayscale
    img = img.resize((28, 28))  # Resize về kích thước 28x28
    img_data = np.array(img).reshape(1, -1) / 255.0  # Chuẩn hóa

    # Dự đoán với BallTree
    dist, ind = tree.query(img_data, k=1)
    predicted_label = train_labels[ind[0][0]]
    return predicted_label

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    image_url = None

    if request.method == 'POST':
        if 'file' not in request.files:
            return "Không tìm thấy file"
        file = request.files['file']
        if file.filename == '':
            return "Tên file không hợp lệ"
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)  # Lưu file tải lên

            # Dự đoán chữ số
            prediction = predict_digit(filepath)
            image_url = url_for('static', filename=f'uploads/{file.filename}')

    return render_template('index.html', prediction=prediction, image_url=image_url)

if __name__ == '__main__':
    # Đảm bảo thư mục lưu trữ ảnh tồn tại
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
