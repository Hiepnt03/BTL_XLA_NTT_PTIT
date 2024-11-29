import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
def sobel_operator(image):
    m,n = img.shape
    gx = [  [-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    gy = [  [ 1, 2, 1], [ 0, 0, 0], [-1,-2,-1]]
    output = np.zeros([m,n])
    # tính tích chập để tìm gradient theo x, gradient theo y, gradient tổng hợp
    for i in range(1, m - 1):
        for j in range(1, n - 1):
            Gx = sum(image[i + k - 1][j + l - 1] * gx[k][l] for k in range(3) for l in range(3))
            Gy = sum(image[i + k - 1][j + l - 1] * gy[k][l] for k in range(3) for l in range(3))
            output[i][j] = int(math.sqrt(Gx ** 2 + Gy ** 2))
    output = output.astype(np.uint8)
    return output

fig = plt.figure(figsize=(10, 4)) # Tạo vùng tỷ lệ 12:7
ax1, ax2 = fig.subplots(1,2) # Tạo 2 vùng vẽ con

# Đọc và hiển thị ảnh gốc
# img = cv2.imread('images/elephant_g.jpg',0)
# img = cv2.imread('images/zebras.jpg',0)
img = cv2.imread('images/hill.jpg',0)
ax1.imshow(img, cmap='gray')
ax1.set_title("Ảnh gốc")
ax1.axis('off')

# Ảnh kết quả
imgSobel = sobel_operator(img)
ax2.imshow(imgSobel, cmap='gray')
ax2.set_title("Ảnh kết quả")
ax2.axis('off')

plt.show()# Hiển thị vùng vẽ

