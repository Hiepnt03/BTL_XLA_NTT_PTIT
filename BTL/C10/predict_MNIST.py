from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Load the saved model for inference
saved_model = load_model('C10\mnist_cnn_model.h5')

# Function to predict a number from an image
def predict_number(image_path):
    # Load and preprocess the image
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to 28x28
    img_array = np.array(img).astype('float32') / 255.0  # Normalize
    img_array = img_array.reshape(1, 28, 28, 1)  # Reshape to match model input
    
    # Predict
    prediction = saved_model.predict(img_array)
    predicted_number = np.argmax(prediction)
    
    return predicted_number

# List of test image paths
test_image_paths = ['C10/images/number3.png', 'C10/images/number5.png', 'C10/images/number7.png']

# Predict digits for each image
for image_path in test_image_paths:
    predicted_digit = predict_number(image_path)
    print(f"The predicted digit for {image_path} is: {predicted_digit}")
