import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


# Load the trained model
model = load_model('model-009.h5')

# Function to load and preprocess the image
def load_preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(256, 256))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0) / 255.0  # Normalize and add batch dimension
    return img

# Function to denoise an image using the model
def denoise_image(model, img):
    predicted_img = model.predict(img)
    return np.squeeze(predicted_img, axis=0)  # Remove batch dimension

# Path to the specific image to process
img_path = 'test_img/6.jpg'  # Update with the correct path

# Directory to save processed images
output_dir = 'result'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load and preprocess the image
img = load_preprocess_image(img_path)

# Use the model to denoise the image
denoised_img = denoise_image(model, img)

# Convert the denoised image to a format that can be saved/displayed
denoised_img_display = (denoised_img * 255).astype(np.uint8)

# Save the result
output_path = os.path.join(output_dir, os.path.basename(img_path))
image.save_img(output_path, denoised_img_display)
print(f"Processed image saved to {output_path}")


