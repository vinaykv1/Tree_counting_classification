

import torch
import cv2
import pandas as pd

# Load the YOLOv5 model (replace 'path/to/your/last.pt' with the actual path to your trained model)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp5/weights/last.pt', force_reload=True)


# Function to count objects in an image
def count_trees_in_image(image_path, output_image_path):
    # Load image
    img = cv2.imread(image_path)

    # Perform inference
    results = model(img)

    # Parse results
    detected_objects = results.pandas().xyxy[0]  # DataFrame with results

    # Filter detections to only include the class for tree stems (assuming class 0 is for tree stems)
    tree_stems = detected_objects[detected_objects['class'] == 0]

    # Count the number of detected tree stems
    tree_count = len(tree_stems)

    print(f'Number of trees detected: {tree_count}')

    # Draw bounding boxes and labels
    for index, row in tree_stems.iterrows():
        xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        confidence = row['confidence']
        class_name = 'Tree'  # Assuming class 0 is 'Tree', replace with actual class name if different
        label = f'{index + 1}'  # Use index + 1 for the label (1, 2, 3, ...)

        # Calculate the center of the bounding box
        center_x = int((xmin + xmax) / 2)
        center_y = int((ymin + ymax) / 2)

        # Get text size for positioning
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        text_w, text_h = text_size

        # Draw rectangle and label on the frame
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)  # Red color for the rectangle

        # Draw class name on top of the bounding box
        class_label = f'{class_name}: {confidence:.2f}'
        class_text_size, _ = cv2.getTextSize(class_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        class_text_w, class_text_h = class_text_size
        cv2.rectangle(img, (xmin, ymin - class_text_h - 10), (xmin + class_text_w, ymin), (0, 255, 0),
                      -1)  # Green background for the text
        cv2.putText(img, class_label, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                    2)  # White color for the text

        # Draw count label in the center of the bounding box
        cv2.rectangle(img, (center_x - text_w // 2 - 5, center_y - text_h // 2 - 5),
                      (center_x + text_w // 2 + 5, center_y + text_h // 2 + 5), (255, 255, 255),
                      -1)  # White background for the text
        cv2.putText(img, label, (center_x - text_w // 2, center_y + text_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 2)  # Black color for the text

    # Save the image with bounding boxes
    cv2.imwrite(output_image_path, img)

    # Display the image with bounding boxes in a window resized to fit the image dimensions
    img_height, img_width = img.shape[:2]
    cv2.namedWindow('Tree Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Tree Detection', img_width, img_height)
    cv2.imshow('Tree Detection', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Example usage
image_path = 'data_8.png'  # Replace this with the actual path to your image file
output_image_path = 'output/output_102.png'  # Replace this with the desired output path for the image with bounding boxes
count_trees_in_image(image_path, output_image_path)
