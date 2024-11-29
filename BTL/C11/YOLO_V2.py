import os
import cv2
import numpy as np

# Đường dẫn tới tệp cấu hình và trọng số của YOLO
cfg_path = "C11/yolov2.cfg"
weights_path = "C11/yolov2.weights" 
names_path = "C11/coco.names" 

# Đảm bảo các ảnh bạn muốn xử lý
image_paths = ["C11/horses.jpg","C11/men_dog.jpg","C11/prhino.jpg"]

output_dir = "results"  # Thư mục để lưu kết quả

# Tạo thư mục lưu kết quả nếu chưa có
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Tải mô hình YOLO v2 từ tệp cấu hình và trọng số
net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)

# Tải các lớp đầu ra từ mạng
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Tải các tên lớp từ tệp coco.names
with open(names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Xử lý từng ảnh
for image_path in image_paths:
    # Đọc ảnh
    img = cv2.imread(image_path)
    
    # Kiểm tra xem ảnh có được tải không
    if img is None:
        print(f"Error loading image {image_path}")
        continue

    # Chuyển ảnh sang dạng blob để xử lý với mạng
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    # Cung cấp blob vào mạng
    net.setInput(blob)

    # Tiến hành nhận diện đối tượng
    detections = net.forward(output_layers)

    # Vẽ các hộp giới hạn lên ảnh (nếu có)
    height, width, channels = img.shape
    for out in detections:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            # Nếu độ tin cậy đủ cao
            if confidence > 0.5:
                # Tính toán vị trí của hộp giới hạn
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                # Tạo các tọa độ góc của hộp giới hạn
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Vẽ hộp giới hạn lên ảnh
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Gắn nhãn đối tượng lên ảnh
                label = str(classes[class_id])
                cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Lưu kết quả với tên mới
    result_image_path = 'C11' + '/' + str(os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_predictions.jpg"))
    cv2.imwrite(result_image_path, img)
    print(f"Result saved to {result_image_path}")

print("All images processed.")
