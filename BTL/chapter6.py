import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("images/fingerprint.jpg", 0)# Đọc ảnh dấu vân tay
_, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)# Chuyển đổi ảnh thành ảnh nhị phân
kernel = np.ones((3, 3), np.uint8)# Chọn phần tử cấu trúc (structuring element)
opening = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)# Áp dụng phép Opening
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)# Áp dụng phép Closing

fig = plt.figure(figsize=(10, 4)) # Tạo vùng tỷ lệ 10:4
ax1, ax2 = fig.subplots(1,2) # Tạo 2 vùng vẽ con
# Hiển thị kết quả
ax1.imshow(image, cmap='gray')
ax1.set_title("Ảnh gốc")
ax1.axis('off')

ax2.imshow(closing, cmap='gray')
ax2.set_title("Opening + Closing")
ax2.axis('off')

plt.show()# Hiển thị vùng vẽ
