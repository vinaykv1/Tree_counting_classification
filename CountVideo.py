import torch
import cv2

# Load the YOLOv5 model (replace 'path/to/your/best.pt' with the actual path to your trained model)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp5/weights/last.pt', force_reload=True)


# Function to count objects in a single frame and draw bounding boxes
def count_trees_and_draw(frame):
    # Perform inference
    results = model(frame)

    # Parse results
    detected_objects = results.pandas().xyxy[0]  # DataFrame with results

    # Filter detections to only include the class for tree stems (assuming class 0 is for tree stems)
    tree_stems = detected_objects[(detected_objects['class'] == 0) & (detected_objects['confidence'] > 0.5)]  # Adjust confidence threshold here

    # Draw bounding boxes and labels
    for index, row in tree_stems.iterrows():
        xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        confidence = row['confidence']
        label = f'Tree: {confidence:.2f}'

        # Draw rectangle and label on the frame
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)  # Red color for bounding box
        cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)  # Red color for text

    # Count the number of detected tree stems
    tree_count = len(tree_stems)

    return tree_count, frame


# Function to process the video and count tree stems in each frame
def count_trees_in_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    total_tree_count = 0

    # Get the original video resolution
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create an output window with the same resolution
    cv2.namedWindow('Tree Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Tree Detection', width, height)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        tree_count, frame_with_boxes = count_trees_and_draw(frame)
        total_tree_count += tree_count

        # Display the frame with bounding boxes and count
        cv2.putText(frame_with_boxes, f'Total Trees: {total_tree_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 2)  # Red color for count display
        cv2.imshow('Tree Detection', frame_with_boxes)

        # Press 'q' to quit the video display
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f'Total number of tree stems detected: {total_tree_count}')


# Example usage
video_path = 'video_1.mp4'  # Replace this with the actual path to your video file
count_trees_in_video(video_path)
