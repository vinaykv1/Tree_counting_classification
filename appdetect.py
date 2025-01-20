# -*- coding: utf-8 -*-
"""Plant Detection and Classification Testing Script

This script loads a pre-trained Keras model for plant trait prediction and utilizes YOLOv8 for plant type detection and counting. It processes new images, predicts plant traits, detects plant types, and counts the frequency of each plant type across the dataset.

## Steps Covered:
1. Install and import necessary libraries.
2. Configure settings and ensure reproducibility.
3. Load the pre-trained Keras model.
4. Initialize the YOLOv8 model for plant type detection.
5. Define functions for image preprocessing, trait prediction, and plant type detection.
6. Process new images to obtain predictions and detections.
7. Aggregate and display the frequency of each plant type detected.

**Note:** Ensure that the `best_model.keras` file is in the working directory. Additionally, replace `'yolov8n.pt'` with the appropriate YOLOv8 model path or name as needed.
"""

# Install necessary libraries
#!pip install -q keras-cv keras tensorflow opencv-python pandas numpy tqdm joblib matplotlib ultralytics

# Import Libraries
import os
import keras_cv
import keras
import tensorflow as tf
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
import joblib
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Print Library Versions for Verification
print("TensorFlow:", tf.__version__)
print("Keras:", keras.__version__)
print("KerasCV:", keras_cv.__version__)
print("YOLO (Ultralytics):", YOLO.__version__)

# Configuration Class
class CFG:
    seed = 42  # Random seed for reproducibility
    image_size = [224, 224]  # Input image size for Keras model
    batch_size = 32  # Batch size for processing images
    yolov8_model_path = 'yolov8n.pt'  # Path to the YOLOv8 model weights

# Reproducibility: Set random seed for consistent results
keras.utils.set_random_seed(CFG.seed)
np.random.seed(CFG.seed)
tf.random.set_seed(CFG.seed)

# Path to the pre-trained Keras model
MODEL_PATH = 'best_model.keras'

# Directory containing new images for testing
NEW_IMAGES_DIR = '/path/to/new/images/'  # Replace with your directory path

# Load the pre-trained Keras model
print("Loading the pre-trained Keras model...")
model = keras.models.load_model(MODEL_PATH, custom_objects={'R2Loss': R2Loss, 'R2Metric': R2Metric})
print("Model loaded successfully.")
model.summary()

# Initialize YOLOv8 model for plant type detection
print("Initializing YOLOv8 model for plant type detection...")
yolo_model = YOLO(CFG.yolov8_model_path)  # Ensure the YOLOv8 model weights are available
print("YOLOv8 model initialized successfully.")

# Define Function to Preprocess Images for Keras Model
def preprocess_image_keras(image_path):
    """
    Preprocesses the image for trait prediction using the Keras model.
    Steps:
    - Read the image from the given path.
    - Resize to the target image size.
    - Normalize pixel values to [0, 1].
    
    Returns:
    - Preprocessed image as a numpy array.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found or unable to read: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (CFG.image_size[1], CFG.image_size[0]), interpolation=cv2.INTER_AREA)
    image = image.astype('float32') / 255.0
    return image

# Define Function to Predict Plant Traits Using Keras Model
def predict_traits(model, image):
    """
    Predicts plant traits using the pre-trained Keras model.
    
    Parameters:
    - model: Loaded Keras model.
    - image: Preprocessed image as a numpy array.
    
    Returns:
    - Trait predictions as a numpy array.
    """
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    preds = model.predict(image)
    trait_preds = preds['head'][0]  # Assuming 'head' is the main task output
    return trait_preds

# Define Function to Detect Plant Types Using YOLOv8
def detect_plant_types(yolo_model, image_path):
    """
    Detects plant types in an image using YOLOv8.
    
    Parameters:
    - yolo_model: Initialized YOLOv8 model.
    - image_path: Path to the image file.
    
    Returns:
    - Detected plant types and their counts as a dictionary.
    - Image with detections (for visualization if needed).
    """
    results = yolo_model(image_path)
    detections = results[0].boxes  # Get detections from the first (and only) image
    labels = detections.cls.cpu().numpy().astype(int)  # Class labels
    class_names = yolo_model.model.names  # Get class names from YOLO model
    detected_classes = [class_names[label] for label in labels]
    counts = pd.Series(detected_classes).value_counts().to_dict()
    return counts, results[0].plot()  # Return counts and image with detections

# Initialize a DataFrame to Hold Results
results_df = pd.DataFrame(columns=['Image_Path', 'Trait_Predictions', 'Plant_Type_Counts'])

# List of All New Image Paths
new_image_paths = glob(os.path.join(NEW_IMAGES_DIR, '*.jpeg'))  # Modify the extension if needed
print(f"Found {len(new_image_paths)} images for testing.")

# Iterate Over Each Image, Predict Traits, and Detect Plant Types
for img_path in tqdm(new_image_paths, desc="Processing Images"):
    try:
        # Preprocess Image for Keras Model
        preprocessed_image = preprocess_image_keras(img_path)
        
        # Predict Plant Traits
        trait_predictions = predict_traits(model, preprocessed_image)
        
        # Detect Plant Types and Count Frequencies
        plant_counts, detection_image = detect_plant_types(yolo_model, img_path)
        
        # Append Results to DataFrame
        results_df = results_df.append({
            'Image_Path': img_path,
            'Trait_Predictions': trait_predictions,
            'Plant_Type_Counts': plant_counts
        }, ignore_index=True)
        
        # (Optional) Save or Display Detection Images
        # detection_image.save(f'detections/{os.path.basename(img_path)}')  # Ensure 'detections' directory exists
        
    except Exception as e:
        print(f"Error processing {img_path}: {e}")

# Display Aggregated Plant Type Frequencies
aggregated_counts = {}
for counts in results_df['Plant_Type_Counts']:
    for plant_type, count in counts.items():
        if plant_type in aggregated_counts:
            aggregated_counts[plant_type] += count
        else:
            aggregated_counts[plant_type] = count

print("\nAggregated Plant Type Frequencies:")
for plant_type, count in aggregated_counts.items():
    print(f"{plant_type}: {count}")

# (Optional) Save Results to CSV
results_df.to_csv('plant_detection_results.csv', index=False)
print("\nResults saved to 'plant_detection_results.csv'.")

# (Optional) Visualize Trait Predictions for the First Few Images
num_samples_to_plot = 5
plt.figure(figsize=(15, num_samples_to_plot * 3))
for i in range(min(num_samples_to_plot, len(results_df))):
    plt.subplot(num_samples_to_plot, 2, 2*i+1)
    img_path = results_df.iloc[i]['Image_Path']
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.title(f"Image: {os.path.basename(img_path)}")
    plt.axis('off')
    
    plt.subplot(num_samples_to_plot, 2, 2*i+2)
    trait_preds = results_df.iloc[i]['Trait_Predictions']
    trait_names = CFG.class_names
    plt.bar(trait_names, trait_preds)
    plt.title("Trait Predictions")
    plt.xlabel("Traits")
    plt.ylabel("Predicted Values")
    plt.xticks(rotation=45)
    
plt.tight_layout()
plt.show()
