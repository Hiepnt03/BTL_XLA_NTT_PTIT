import numpy as np
from skimage import io, color, util, filters
import matplotlib.pyplot as plt

# Hàm tính năng lượng của ảnh
def compute_energy(image):
    gray_image = color.rgb2gray(image)
    energy_map = filters.sobel(gray_image)
    return energy_map

# Hàm tính toán seam có năng lượng thấp nhất
def find_vertical_seam(energy_map):
    rows, cols = energy_map.shape
    seam = np.zeros(rows, dtype=np.int32)
    cumulative_energy = energy_map.copy()
    
    # Tính năng lượng tích lũy
    for i in range(1, rows):
        for j in range(cols):
            left = cumulative_energy[i-1, j-1] if j > 0 else float('inf')
            up = cumulative_energy[i-1, j]
            right = cumulative_energy[i-1, j+1] if j < cols-1 else float('inf')
            cumulative_energy[i, j] += min(left, up, right)
    
    # Truy vết seam
    seam[-1] = np.argmin(cumulative_energy[-1])
    for i in range(rows-2, -1, -1):
        prev_col = seam[i+1]
        offsets = [-1, 0, 1]
        valid_offsets = [offset for offset in offsets if 0 <= prev_col + offset < cols]
        seam[i] = prev_col + min(valid_offsets, key=lambda offset: cumulative_energy[i, prev_col + offset])
    
    return seam

# Hàm loại bỏ một seam
def remove_vertical_seam(image, seam):
    rows, cols, _ = image.shape
    new_image = np.zeros((rows, cols-1, 3))
    for i in range(rows):
        col = seam[i]
        new_image[i, :, 0] = np.delete(image[i, :, 0], col)
        new_image[i, :, 1] = np.delete(image[i, :, 1], col)
        new_image[i, :, 2] = np.delete(image[i, :, 2], col)
    return new_image

# Hàm thực hiện seam carving
def seam_carve(image, num_seams):
    carved_image = image.copy()
    for _ in range(num_seams):
        energy_map = compute_energy(carved_image)
        seam = find_vertical_seam(energy_map)
        carved_image = remove_vertical_seam(carved_image, seam)
    return carved_image

# path = 'images/swans.jpg'
# path = 'images/man.jpg'
path = 'images/fish.jpg'
# Đọc ảnh
image = io.imread(path)

# Chuyển đổi ảnh về kiểu float
image = util.img_as_float(image)

# Thực hiện seam carving
image_seam_carved = seam_carve(image, 50)

# Hiển thị kết quả
fig = plt.figure(figsize=(12, 7)) # Tạo vùng tỷ lệ 12:7
(ax1, ax2) = fig.subplots(1,2) # Tạo 2 vùng vẽ con

ax1.imshow(image)
ax1.set_title("Original Image")
ax1.axis('off')
ax2.imshow(image_seam_carved)
ax2.set_title("Seam Carved Image")
ax2.axis('off')
plt.show()
