import cv2
import torch
from ultralytics import YOLO
import numpy as np
import os
import time

# Load YOLO Pose model
pose_model = YOLO("yolo11n-pose.pt")  # Pose estimation model

# Directories
input_dir = r"C:\code\emb\codew\new_in"
output_dir = r"C:\code\emb\codew\out_t"
screenshot_dir = os.path.join(output_dir, "screenshots")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(screenshot_dir, exist_ok=True)

# Processing settings
skip_frames = 1
hand_stay_time = 2  # seconds before suspicion
hand_timers = {}

# COCO Pose Keypoints
LEFT_WRIST = 9
RIGHT_WRIST = 10
NOSE = 0  # Used for front/back detection

# Process videos
for file_name in os.listdir(input_dir):
    file_path = os.path.join(input_dir, file_name)
    output_path = os.path.join(output_dir, f"processed_{file_name}")

    if not file_name.lower().endswith(('.mp4', '.avi', '.mov')):
        continue

    cap = cv2.VideoCapture(file_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    frame_rate = int(cap.get(5))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (frame_width, frame_height))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % skip_frames == 0:
            pose_results = pose_model(frame)
            suspicious_detected = False

            for result in pose_results:
                keypoints = result.keypoints.xy.cpu().numpy() if result.keypoints is not None else []
                boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else []

                for i in range(len(boxes)):
                    x1, y1, x2, y2 = map(int, boxes[i])
                    width = x2 - x1
                    height = y2 - y1
                    
                    # Chest Box (Shirt/Hoodie)
                    chest_box = [x1 + int(0.1 * width), y1, x2 - int(0.1 * width), y1 + int(0.4 * height)]
                    
                    # Waist Boxes (Pant Pockets)
                    left_waist_box = [x1, y1 + int(0.5 * height), x1 + int(0.5 * width), y2]
                    right_waist_box = [x1 + int(0.5 * width), y1 + int(0.5 * height), x2, y2]
                    
                    # Draw Chest and Waist Boxes
                    cv2.rectangle(frame, tuple(chest_box[:2]), tuple(chest_box[2:]), (0, 255, 255), 2)  # Yellow
                    cv2.rectangle(frame, tuple(left_waist_box[:2]), tuple(left_waist_box[2:]), (255, 0, 0), 2)  # Blue
                    cv2.rectangle(frame, tuple(right_waist_box[:2]), tuple(right_waist_box[2:]), (0, 0, 255), 2)  # Red
                    
                    if len(keypoints) <= i:
                        continue

                    person_keypoints = keypoints[i]
                    left_wrist = person_keypoints[LEFT_WRIST] if person_keypoints[LEFT_WRIST][0] > 0 else None
                    right_wrist = person_keypoints[RIGHT_WRIST] if person_keypoints[RIGHT_WRIST][0] > 0 else None
                    nose = person_keypoints[NOSE] if person_keypoints[NOSE][0] > 0 else None  # For front/back detection
                    
                    # Hand Tracking Boxes
                    if left_wrist is not None:
                        lw_x, lw_y = int(left_wrist[0]), int(left_wrist[1])
                        cv2.rectangle(frame, (lw_x - 15, lw_y - 15), (lw_x + 15, lw_y + 15), (0, 255, 0), 2)
                        cv2.putText(frame, "Left Hand", (lw_x, lw_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    if right_wrist is not None:
                        rw_x, rw_y = int(right_wrist[0]), int(right_wrist[1])
                        cv2.rectangle(frame, (rw_x - 15, rw_y - 15), (rw_x + 15, rw_y + 15), (0, 255, 0), 2)
                        cv2.putText(frame, "Right Hand", (rw_x, rw_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Checking if the person is facing away
                    is_back_facing = nose is None or nose[1] < y1  # Nose is not visible or positioned high

                    # Checking for suspicious hand positions
                    for wrist, label in [(left_wrist, "left"), (right_wrist, "right")]:
                        if wrist is None:
                            continue
                        
                        wrist_x, wrist_y = int(wrist[0]), int(wrist[1])
                        
                        # Check if hand is inside any bounding box
                        in_chest = chest_box[0] <= wrist_x <= chest_box[2] and chest_box[1] <= wrist_y <= chest_box[3]
                        in_left_waist = left_waist_box[0] <= wrist_x <= left_waist_box[2] and left_waist_box[1] <= wrist_y <= left_waist_box[3]
                        in_right_waist = right_waist_box[0] <= wrist_x <= right_waist_box[2] and right_waist_box[1] <= wrist_y <= right_waist_box[3]

                        if in_chest or in_left_waist or in_right_waist:
                            if label not in hand_timers:
                                hand_timers[label] = time.time()
                            elif time.time() - hand_timers[label] > hand_stay_time:
                                suspicious_detected = True
                                cv2.putText(frame, "⚠️ Suspicious!", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                hand_timers[label] = time.time()
                        else:
                            if label in hand_timers:
                                del hand_timers[label]
                    
                    # If the person is back-facing and hands are missing for too long
                    if is_back_facing and (left_wrist is None or right_wrist is None):
                        if "back_facing" not in hand_timers:
                            hand_timers["back_facing"] = time.time()
                        elif time.time() - hand_timers["back_facing"] > hand_stay_time:
                            suspicious_detected = True
                            cv2.putText(frame, "⚠️ Suspicious (Back-Facing)", (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    else:
                        if "back_facing" in hand_timers:
                            del hand_timers["back_facing"]
                    
            if suspicious_detected:
                screenshot_filename = os.path.join(screenshot_dir, f"suspicious_{file_name}_{frame_count}.jpg")
                cv2.imwrite(screenshot_filename, frame)

            out.write(frame)
            cv2.imshow("Hand-to-Clothing Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_count += 1

    cap.release()
    out.release()

cv2.destroyAllWindows()
print("✅ Processing complete!")
