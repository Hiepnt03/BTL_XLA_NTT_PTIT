import cv2
import numpy as np
from matplotlib import pyplot as plt
# Định nghĩa hàm biến đổi DFT
def DFT1D(img):
    U = len(img)
    outarry = np.zeros(U, dtype=complex)
    for m in range(U):
        sum = 0.0
        for n in range(U):
            e = np.exp(-1j * 2 * np.pi * m * n / U)
            sum += img[n] * e
        outarry[m] = sum
    return outarry

def IDFT1D(img):
    U = len(img)
    outarry = np.zeros(U, dtype=complex)
    for n in range(U):
        sum = 0.0
        for m in range(U):
            e = np.exp(1j * 2 * np.pi * m * n / U)
            sum += img[m] * e
        pixel = sum / U
        outarry[n] = pixel
    return outarry

#Định nghĩa hàm lọc thông thấp ideals
def lowPass_Ideals(D0, U, V):
    # H là bộ lọc
    H = np.zeros((U, V))
    D = np.zeros((U, V))
    U0 = int(U / 2)
    V0 = int(V / 2)
    
    # Tính khoảng cách 
    for u in range(U):
        for v in range(V):
            u2 = np.power(u, 2)
            v2 = np.power(v, 2)
            D[u, v] = np.sqrt(u2 + v2)
    # Tính bộ lọc
    for u in range(U):
        for v in range(V):
            if D[np.abs(u - U0), np.abs(v - V0)] <= D0:
                H[u, v] = 1
            else:
                H[u, v] = 0
    return H

if __name__ == "__main__":
    # đọc ảnh
    # path = 'images/beans_g.png'
    # path = 'images/cameraman.jpg'
    path = 'images/elephant_g.jpg'
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(src=img, dsize=(100, 100))
    # Chuyển các pixel của ảnh vào mảng 2 chiều
    f = np.asarray(img)
    M,N = np.shape(f) # chiều x và y cuả ảnh
    
    # Bước 1: Chuyển ảnh từ kích thước MxN thành PxQ với P = 2M  và Q = 2N
    P, Q = 2*M, 2*N
    shape = np.shape(f)
    # Chuyển ảnh PxQ vào mảng fp
    f_xy_p = np.zeros((P, Q))
    f_xy_p[:shape[0], :shape[1]] = f
    
    # Bước 2: Nhân ảnh f_xy_p với (-1)^(x+y)
    F_xy_p = np.zeros((P, Q))
    for x in range(P):
        for y in range(Q):
            F_xy_p[x, y] = f_xy_p[x, y] * np.power(-1, x + y)
            
    # Bước 3: Chuyển đổi ảnh Fpc sang miền tần số DFT
    dft_cot = dft_hang = np.zeros((P, Q))
    # DFT theo cột
    for x in range(P):
        dft_cot[x] = DFT1D(F_xy_p[x])
    # DFT theo hàng
    for y in range(Q):
        dft_hang[:, y] = DFT1D(dft_cot[:, y])
        
    # Bước 4: Lọc thông thấp ideals
    H_uv = lowPass_Ideals(60, P, Q)
    
    # Bước 5: Nhân ảnh sau khi DFT với ảnh sau khi lọc
    G_uv = np.multiply(dft_hang, H_uv)
    
    # Bước 6
    # Bước 6.1: Thực hiện biến đổi ngược DFT
    idft_cot = idft_hang = np.zeros((P, Q))
    for x in range(P):
        idft_cot[x] = IDFT1D(G_uv[x])
    for y in range(Q):
        idft_hang[:, y] = IDFT1D(idft_cot[:, y])
    
    # Bước 6.2: Nhân phần thực ảnh sau khi biến đổi ngược với (-1)^(x+y)
    g_array = np.asarray(idft_hang.real)
    P,Q = np.shape(g_array)
    g_xy_p = np.zeros((P, Q))
    for x in range(P):
        for y in range(Q):
            g_xy_p[x, y] = g_array[x, y] * np.power(-1, x + y)
    
    # Bước 7: Rút trích ảnh kích thước MxN từ PxQ  
    g_xy = g_xy_p[:shape[0], :shape[1]]
    
    # Hiển thị ảnh
    fig = plt.figure(figsize=(12, 7)) # Tạo vùng tỷ lệ 12:7
    (ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9) = fig.subplots(3,3) # Tạo 3 cột 3 hàng
    
    img2 = cv2.imread(path)
    ax1.imshow(img2, cmap='gray')
    ax1.set_title("Ảnh gốc")
    ax1.axis('off')
    
    ax2.imshow(f_xy_p, cmap='gray')
    ax2.set_title("Bước 1: Ảnh PxQ")
    ax2.axis('off')
    
    ax3.imshow(F_xy_p, cmap='gray')
    ax3.set_title("Bước 2: Nhân ảnh f_xy_p với (-1)^(x+y)")
    ax3.axis('off')
    
    ax4.imshow(dft_hang, cmap='gray')
    ax4.set_title("Bước 3: Phổ tần số ảnh sau khi DFT")
    ax4.axis('off')
    
    ax5.imshow(H_uv, cmap='gray')
    ax5.set_title("Bước 4: Phổ tần số ảnh bộ lọc")
    ax5.axis('off')
    
    ax6.imshow(G_uv, cmap='gray')
    ax6.set_title("Bước 5: Phổ tần số ảnh sau khi lọc")
    ax6.axis('off')
    
    ax7.imshow(idft_hang, cmap='gray')
    ax7.set_title("Bước 6.1: DFT ngược")
    ax7.axis('off')
    
    ax8.imshow(g_xy_p, cmap='gray')
    ax8.set_title("Bước 6.2: Nhân với (-1)^(x+y)")
    ax8.axis('off')
    
    ax9.imshow(g_xy, cmap='gray')
    ax9.set_title("Bước 7: Ảnh kết quả")
    ax9.axis('off')
    
    plt.show() # Hiển thị vùng vẽ