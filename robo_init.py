from rtde_control import RTDEControlInterface as RTDEControl
import threading
import time
import queue
import datetime
import cv2
import pykinect_azure as pykinect

pykinect.initialize_libraries()

# Modify camera configuration
device_config = pykinect.default_configuration
device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1536P
device_config.depth_mode = pykinect.K4A_DEPTH_MODE_NFOV_2X2BINNED

# Start device
device = pykinect.start_device(config=device_config)

from yolo_model_init import run_yolo 
# cv2 Setup: VideoCapture and Window creation
window = cv2.namedWindow('Human Detection', cv2.WINDOW_NORMAL)
window = cv2.resizeWindow('Human Detection', 1024, 768)

# FPS list for each frame for avg FPS calculation

# Initialize RTDE Control
rtde_c = RTDEControl("172.26.248.89")

# Parameters
acceleration = 0.5
initial_joint_q = [-1.54, -1.83, -2.28, -0.59, 1.60, 0.023]
end_joint_q = [-0.5, -1.0, -1.0, -1.0, 1.5, 0.5]

# Speed levels based on distance ranges
speed_levels = [0.0, 0.2, 0.45, 0.7, 1.0]
max_speed = 1

# Distance thresholds (in meters)
distance_thresholds = [600, 1000, 1500, 2500]

# Queue for notifying speed changes
speed_queue = queue.Queue()


def robot_motion():
    current_speed_level = 1
    current_speed = speed_levels[current_speed_level] * max_speed
    try:
        while True:

            # Check for speed updates
            try:
                new_speed_level = speed_queue.get_nowait()
                if new_speed_level != current_speed_level:
                    current_speed_level = new_speed_level
                    current_speed = speed_levels[current_speed_level] * max_speed
                    print(f"Speed updated to {current_speed}")

            except queue.Empty:
                pass

            if current_speed == 0:
                continue

            # Move to end position and back to initial position
            rtde_c.moveJ(end_joint_q, speed=current_speed,
                         acceleration=acceleration)

            try:
                new_speed_level = speed_queue.get_nowait()
                if new_speed_level != current_speed_level:
                    current_speed_level = new_speed_level
                    current_speed = speed_levels[current_speed_level] * max_speed
                    print(f"Speed updated to {current_speed}")

            except queue.Empty:
                pass

            if current_speed == 0:
                continue

            rtde_c.moveJ(initial_joint_q, speed=current_speed,
                         acceleration=acceleration)

    finally:
        rtde_c.speedStop()
        rtde_c.stopScript()
        print("Motion stopped, script terminated.")


def depth_detection():
    print("depththread started")
    global depth_distance
    current_range = -1
    fps_list = []
    while True:
        start = datetime.datetime.now()  # Start time for FPS calculation

        # Azure Kinect frame capture
        capture = device.update()
        ret, frame = capture.get_color_image()
        success, depth_frame = capture.get_depth_image()
        if not ret or not success:  # break if no frame
            break

        # resize frame
        frame = cv2.resize(frame, (1024, 768))

        # MODEL SELECTION
        # Comment out the model not in use

        # YOLO detection
        frame, depth_distance = run_yolo(frame, depth_frame)

        # Mediapipe detection
        # frame = run_mediapipe(frame, depth_frame)

        end = datetime.datetime.now()  # end time
        total = (end - start).total_seconds()  # processing time

        # FPS
        fps = 1 / total
        fps_list.append(fps)
        cv2.putText(frame, f"FPS: {fps:.2f}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8)

        # show depth frame (comment out when testing
        cv2.imshow("depth", depth_frame)
        # show frame (comment out when testing)
        cv2.imshow("Human Detection", frame)

        # Determine which range the distance falls into
        if depth_distance < distance_thresholds[0]:
            new_range = 0
        elif depth_distance < distance_thresholds[1]:
            new_range = 1
        elif depth_distance < distance_thresholds[2]:
            new_range = 2
        elif depth_distance < distance_thresholds[3]:
            new_range = 3
        else:
            new_range = 4

        # If the range has changed, notify the robot control thread
        if new_range != current_range:
            current_range = new_range
            speed_queue.put(new_range)

        if cv2.waitKey(1) == ord("q"):  # force quit
            break

# Start the depth detection and robot motion threads


motion_thread = threading.Thread(target=robot_motion)


motion_thread.start()

depth_detection()

motion_thread.join()
