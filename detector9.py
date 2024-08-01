#For the tracking
import cv2 as cv
import numpy as np
import pandas as pd
from ultralytics import YOLO
import sys
import os
 
sys.path.append(os.path.abspath('D:/FutureExpertData/Computervision/sort'))
from sort import Sort
import time

def line_angle(x1, y1, x2, y2):
    return np.degrees(np.arctan2(y2 - y1, x2 - x1))

def extract_lines(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    edges = cv.Canny(blurred, 50, 150, apertureSize=3)
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, threshold=150, minLineLength=50, maxLineGap=10)
    return lines

def extract_frame(video, frame_number):
    if not video.isOpened():
        print("Error: Could not open video.")
        exit()
    video.set(cv.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = video.read()
    if ret:
        cv.imwrite('frame_5000èyh.jpg', frame)
    else:
        print(f"Error: Could not retrieve frame {frame_number}.")
    video.set(cv.CAP_PROP_POS_FRAMES, 0)
    return frame

def detect_line_areas(df, lines, x1_tank, y1_tank, x2_tank, y2_tank):
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = line_angle(x1, y1, x2, y2)
            if -7 <= angle <= 7:
                if (x1 > x1_tank and x2 < x2_tank) and (y1 > y1_tank and y2 < y2_tank):
                    df = pd.concat([df, pd.DataFrame([{"x1": x1, "y1": y1, "x2": x2}])], ignore_index=True)
    df_sorted = df.sort_values(by="y1")
    yy = 0
    first_row = df_sorted['y1'].iloc[0]
    last_row = df_sorted['y1'].iloc[-1]
    j = (last_row - first_row) / 4
    areas = pd.DataFrame(columns=['x1', 'y1', 'x2', 'y2'])
    for index, row in df_sorted.iterrows():
        if np.abs(row['y1'] - yy) > j:
            yy = row['y1']
            areas = pd.concat([areas, df_sorted[df_sorted['y1'] == yy]], ignore_index=True)
    areas = areas.sort_values(by="y1")
    return areas

def detect_blue_cap(frame, x1_tank, width):
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    lower_blue = np.array([100, 150, 0])
    upper_blue = np.array([140, 255, 255])
    mask = cv.inRange(hsv, lower_blue, upper_blue)

    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    positions = []

    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        if w * h > 500 and (x > x1_tank and w + x < width - x1_tank):
            positions.append((x, y, w, h))

    return frame, positions

model = YOLO("D:/FutureExpertData/Computervision/best.pt")
video = cv.VideoCapture("D:/FutureExpertData/Computervision/cortado_laberinto2.MP4")

fish_zones = {}
zone_detection_counter = {}
zones_written = {}

fps = video.get(cv.CAP_PROP_FPS)
typical_occlusion_time = 1000  # seconds
max_age = int(fps * typical_occlusion_time)  # Calculate max_age based on fps and occlusion time

tracker = Sort(max_age=max_age)  # Initialize SORT with calculated max_age

id_colors = {
    0: (255, 0, 0),   # Red
    1: (0, 255, 0),   # Green
    2: (0, 0, 255),   # Blue
    3: (255, 255, 0), # Cyan
    4: (255, 0, 255), # Magenta
    5: (0, 255, 255), # Yellow
    6: (128, 0, 0),   # Maroon
    7: (0, 128, 0),   # Dark Green
    8: (0, 0, 128),   # Navy
    9: (128, 128, 0)  # Olive
}

with open("zone_transitions.txt", "w") as zones_file, \
     open("lines_positions.txt", "w") as lines_file, \
     open("boxes_positions.txt", "w") as boxes_file, \
     open("bottle_position.txt", "w") as bottle_file:

    nframe = extract_frame(video, 500)
    lines = extract_lines(nframe)

    index_frame = 0
    box_limit = 0
    limit=0

    while True:
        ret, frame = video.read()
        if not ret:
            break

        height, width, channels = frame.shape
        x1_tank = width / 4
        y1_tank = height / 12
        x2_tank, y2_tank = width - x1_tank, height - y1_tank

        results = model.track(frame, persist=True)
        boxes = results[0].boxes

        lines_file.write(f"Frame {index_frame} \n")
        df = pd.DataFrame(columns=['x1', 'y1', 'x2', 'y2'])
        areas = detect_line_areas(df, lines, x1_tank, y1_tank, x2_tank, y2_tank)
        for index, area in areas.iterrows():
            lines_file.write(f"Line {index}, Coordinates : ({area['x1']}, {area['y1']}, {area['x2']}, {area['y2']})\n")
        lines_file.write(f"\n")

        frame, positions = detect_blue_cap(frame, x1_tank, width)
        for pos in positions:
            y_botella = pos[3] + pos[1]
            if index_frame == 1000 and box_limit == 0:
                bottle_file.write(f"Position: x={pos[0]}, y={pos[1]}, width={pos[2]}, height={pos[3]}, y_botella={y_botella}\n")
                box_limit = 1

        boxes_file.write(f"Frame {index_frame} \n")

        detections = []
        for idx, box in enumerate(boxes):
            if idx >= 5:
                break
            xmin, ymin, xmax, ymax = box.xyxy[0].tolist()
            confidence = box.conf[0].item()
            if confidence > 0.55:
                detections.append([xmin, ymin, xmax, ymax, confidence])

        tracked_objects = tracker.update(np.array(detections))

        for obj in tracked_objects:
            x1, y1, x2, y2, obj_id = [int(v) for v in obj]
            color = id_colors.get(obj_id % len(id_colors), (255, 255, 255))
            cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv.putText(frame, f'ID: {obj_id}', (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            boxes_file.write(f"ID: {obj_id}, BBox: ({x1}, {y1}, {x2}, {y2})\n")

            # (Here you can continue with your zone and fish tracking logic as before)

       # print(f"{time:.2f}s \n")
        boxes_file.write(f"\n")
        index_frame += 1
        frame = cv.resize(frame, (880, 624))
        cv.imshow('Lines', frame)
        if cv.waitKey(1) & 0xFF == ord('q') or limit == 1:
            break

video.release()
cv.destroyAllWindows()
