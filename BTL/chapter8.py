import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, exposure
# Tải hình ảnh xám đầu vào
# path = 'images/tigers.jpeg'
# path = 'images/parrot.jpg'
path = 'images/horse.jpg'
image = io.imread(path, as_gray=True)

# Hiển thị hình ảnh gốc
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.axis('off')

# Tính ngưỡng Otsu
otsu_threshold = filters.threshold_otsu(image)
# Áp dụng ngưỡng để phân đoạn hình ảnh
binary_image = image > otsu_threshold

# Hiển thị biểu đồ histogram với ngưỡng
plt.subplot(1, 3, 2)
plt.title('Histogram and Otsu Threshold')
hist, bins = exposure.histogram(image)
plt.plot(bins, hist, label='Histogram')
plt.axvline(otsu_threshold, color='red', linestyle='--', label=f'Otsu Threshold = {otsu_threshold:.2f}')
plt.legend()

# Hiển thị hình ảnh nhị phân phân đoạn
plt.subplot(1, 3, 3)
plt.title('Segmented Image (Binary)')
plt.imshow(binary_image, cmap='gray')
plt.axis('off')
# Hiển thị hình ảnh
plt.tight_layout()
plt.show()
