import cv2
import numpy as np
import matplotlib.pyplot as plt

def gaussian_filter_frequency_domain(image_path, sigma=10):
    # Đọc ảnh đầu vào ở dạng grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Kích thước của ảnh
    rows, cols = image.shape
    # Biến đổi Fourier
    dft = np.fft.fft2(image)
    dft_shifted = np.fft.fftshift(dft)  # Dịch tần số 0 vào giữa
    # Tạo mặt nạ lọc Gauss trong miền tần số
    x = np.arange(-cols // 2, cols // 2)
    y = np.arange(-rows // 2, rows // 2)
    X, Y = np.meshgrid(x, y)
    gaussian_mask = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    # Áp dụng mặt nạ lọc Gauss
    filtered_dft = dft_shifted * gaussian_mask
    # Biến đổi Fourier ngược để lấy lại ảnh trong miền không gian
    dft_inverse_shifted = np.fft.ifftshift(filtered_dft)
    filtered_image = np.fft.ifft2(dft_inverse_shifted)
    filtered_image = np.abs(filtered_image)  # Lấy phần thực của ảnh
    # Hiển thị ảnh gốc và ảnh đã lọc
    fig = plt.figure(figsize=(12, 7)) # Tạo vùng tỷ lệ 12:7
    (ax1, ax2) = fig.subplots(1,2) # Tạo 2 cột 1 hàng
    ax1.imshow(image,cmap='gray')
    ax1.set_title("Ảnh gốc")
    ax1.axis('off')
    ax2.imshow(np.uint8(filtered_image),cmap='gray')
    ax2.set_title("Kết quả sau khi lọc")
    ax2.axis('off')
    plt.show()
# path = 'images/cameraman.jpg'
# path = 'images/elephant_g.jpg'
path = 'images/mri_T1.png'
filtered_image = gaussian_filter_frequency_domain(path, sigma=20)
