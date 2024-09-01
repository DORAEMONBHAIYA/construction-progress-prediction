import cv2
import numpy as np

def process_image(image_path):
    # Load and preprocess the image
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))  # Resize to a fixed size
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = np.mean(image) / 255.0  # Simple feature extraction (mean pixel value)
    return feature
