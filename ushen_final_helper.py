#  Global imports, functions and variables
import cv2
import numpy as np

CONFIDENCE_THRESHOLD = 0.4
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
RED = (0, 0, 255)


def process_depth(box_center, image, depth_image, locations):
    depth_data = 10000
    if box_center is None or locations is None:
        print("No box detected")
        return image, depth_data

    # get the lowest value in the bounding box that's not 0
    x, y, w, h = locations

    roi = depth_image[y:y+h, x:x+w]

    valid_pixels = roi[roi != 0]

    if valid_pixels.size == 0:
        print("No valid pixels, everything is 0")
        return image, depth_data

    depth_data = np.min(valid_pixels)
    # print(f"Depth data: {depth_data}")

    cv2.putText(image, f"Depth: {depth_data}mm",
                (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, RED, 2)

    min_depth_loc = np.unravel_index(
        np.argmin(np.where(roi == depth_data, roi, np.inf)), roi.shape)
    min_depth_point = (min_depth_loc[1] + x, min_depth_loc[0] + y)

    cv2.circle(image, min_depth_point, radius=5,
               color=(0, 0, 255), thickness=-1)

    # print(min_depth_point)

    return image, depth_data


# YOLO Model
from ultralytics import YOLO
model = YOLO("./models/yolov8n.pt")


def visualize_YOLO(image, detections):
    box_center = None
    locations = None
    for data in detections.boxes.data.tolist():
        confidence = data[4]

        if float(confidence) < CONFIDENCE_THRESHOLD:
            continue

        xmin, ymin, xmax, ymax = int(data[0]), int(
            data[1]), int(data[2]), int(data[3])
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), GREEN, 2)

        box_center = (int((xmin + xmax) // 2),
                      int((ymin + ymax) // 2)-(ymax-ymin)//4)

        locations = [xmin, ymin, xmax, ymax]

    return image, box_center, locations


def get_yolo_detections(frame):
    return model(frame, classes=0, verbose=False)[0]


def run_yolo(frame):
    detections = get_yolo_detections(frame)
    return visualize_YOLO(frame, detections)

