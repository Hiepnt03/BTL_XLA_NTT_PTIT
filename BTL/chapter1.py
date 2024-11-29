import cv2
import numpy as np
import matplotlib.pyplot as plt
    # Bước 1: Đọc ảnh màu
    # path = 'images/zebras.jpg'
    # path = 'images/bisons.jpg'
path = 'images/aero.jpg'
img = cv2.imread(path, cv2.IMREAD_COLOR)
    # Bước 2: Chuyển ảnh màu thành mảng
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Chuyển BGR sang RGB để sử dụng đúng kênh màu
    # Bước 3: Tính Gray = 0.299 x Red + 0.587 x Green + 0.114 x Blue
    # Tách kênh màu R, G, B từ ảnh
red = img_rgb[:, :, 0]
green = img_rgb[:, :, 1]
blue = img_rgb[:, :, 2]
    # Tính giá trị Gray bằng công thức
gray_array = 0.299 * red + 0.587 * green + 0.114 * blue
    # Bước 4: Chuyển mảng Gray thành ảnh
gray_image = gray_array.astype(np.uint8)
# Hiển thị ảnh 
fig = plt.figure(figsize=(12, 7)) # Tạo vùng tỷ lệ 12:7
(ax1, ax2) = fig.subplots(1,2) # Tạo 2 vùng vẽ con
image = cv2.imread(path,cv2.IMREAD_COLOR)
img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
ax1.imshow(img_rgb, cmap='gray')
ax1.set_title("Ảnh gốc")
ax1.axis('off')
ax2.imshow(gray_image, cmap='gray')
ax2.set_title("Ảnh xám")
ax2.axis('off')
plt.show() # Hiển thị vùng vẽ
