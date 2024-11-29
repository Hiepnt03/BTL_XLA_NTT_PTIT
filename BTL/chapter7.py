from skimage.feature import hog
from skimage import exposure
from skimage.color import rgb2gray
from skimage.io import imread
import matplotlib.pyplot as plt
import cv2
# path = 'images/beans_g.png'
# path = 'images/cameraman.jpg'
path = 'images/elephant_g.jpg'
# Đọc ảnh và chuyển đổi sang ảnh xám:
image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
# Tính toán HOG và ảnh HOG:
fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True)
# Hiển thị ảnh gốc và ảnh HOG:
fig, (axes1, axes2) = plt.subplots(1, 2, figsize=(12, 7), sharex=True, sharey=True)
axes1.axis('off'), axes1.imshow(image, cmap=plt.cm.gray)
axes1.set_title('Ảnh gốc')
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
axes2.axis('off'), axes2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
axes2.set_title('Ảnh HOG')
plt.show()