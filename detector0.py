import cv2 as cv
import numpy as np
import pandas as pd
from ultralytics import YOLO
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
        cv.imwrite('frame_1.jpg', frame)
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
                    df = pd.concat([df, pd.DataFrame([{"x1": x1, "y1": y1, "x2": x2, "y2": y2}])], ignore_index=True)
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
    # Detect the blue cap (bottle) using color segmentation
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    lower_blue = np.array([100, 150, 0])
    upper_blue = np.array([140, 255, 255])
    mask = cv.inRange(hsv, lower_blue, upper_blue)
 
    # Find contours for the detected blue areas
    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
 
    # List to store positions of bounding boxes
    positions = []
 
    for contour in contours:
        # Compute the bounding box for the contour
        x, y, w, h = cv.boundingRect(contour)
        # Filter small contours
        if w * h > 500 and (x > x1_tank and w + x < width - x1_tank):
            # Store the position of the bounding box
            positions.append((x, y, w, h))
 
    return frame, positions
 
# Load the YOLO model and video
model = YOLO("D:/FutureExpertData/Computervision/best.pt")

# Open the video file
video = cv.VideoCapture("D:/FutureExpertData/Computervision/chrisus.MP4")
 
fish_zones = {}
zone_detection_counter = {}
zones_written = {}
 
fps = video.get(cv.CAP_PROP_FPS)
limit =0
with open("zone_transitions.txt", "w") as zones_file, \
     open("lines_positions.txt", "w") as lines_file, \
     open("boxes_positions.txt", "w") as boxes_file, \
     open("bottle_position.txt", "w") as bottle_file:
   
    #nframe = extract_frame(video, 10)
    nframe = cv.imread('D:/FutureExpertData/Computervision/frame_5000.jpg')
    lines = extract_lines(nframe)
 
    index_frame = 0
    box_limit = 0
   
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
            #cv.line(frame, (area['x1'], area['y1']), (area['x2'], area['y2']), (0, 255, 0), 2)
        lines_file.write(f"\n")
       
       
        frame, positions = detect_blue_cap(frame, x1_tank, width)
       
        for pos in positions:
            y_botella = pos[3] + pos[1]
            if index_frame==1000 and box_limit == 0:
                bottle_file.write(f"Position: x={pos[0]}, y={pos[1]}, width={pos[2]}, height={pos[3]}, y_botella={y_botella}\n")
                box_limit = 1
 
        boxes_file.write(f"Frame {index_frame} \n")
        for idx, box in enumerate(boxes):
            if idx >= 5:
                break
            xmin, ymin, xmax, ymax = box.xyxy[0].tolist()
            confidence = box.conf[0].item()
            center_y = (ymin + ymax) / 2
            boxes_file.write(f"ID: {idx}, BBox: ({xmin}, {ymin}, {xmax}, {ymax}), Confidence: {confidence:.2f}\n")
           
            if confidence > 0.55:
                try:
                    current_zone = 0
                    x_tank = (x2_tank + x1_tank) / 2
                    if y_botella > height / 2:
                        if len(areas) >= 4:
                            if center_y > 0 and center_y <= areas['y1'].iloc[0]:
                                current_zone = 1
                            elif (center_y > areas['y1'].iloc[0] and center_y <= areas['y1'].iloc[1]) and xmax < x_tank:
                                current_zone = 2
                            elif (center_y > areas['y1'].iloc[1] and center_y <= areas['y1'].iloc[2]) and xmax > x_tank:
                                current_zone = 3
                            elif (center_y > areas['y1'].iloc[2] and center_y <= areas['y1'].iloc[3]) and xmax < x_tank:
                                current_zone = 4
                            elif (center_y > areas['y1'].iloc[3]) and xmax > x_tank:
                                current_zone = 5
                    else:
                        if len(areas) >= 4:
                            if center_y < height and center_y >= areas['y1'].iloc[3]:
                                current_zone = 1
                            elif (center_y < areas['y1'].iloc[3] and center_y >= areas['y1'].iloc[2]) and xmax > x_tank:
                                current_zone = 2
                            elif (center_y < areas['y1'].iloc[2] and center_y >= areas['y1'].iloc[1]) and xmax < x_tank:
                                current_zone = 3
                            elif (center_y < areas['y1'].iloc[1] and center_y >= areas['y1'].iloc[0]) and xmax > x_tank:
                                current_zone = 4
                            elif (center_y < areas['y1'].iloc[0]+100) and xmax < x_tank:
                                current_zone = 5
 
                    fish_id = idx
                    if fish_id not in zone_detection_counter:
                        zone_detection_counter[fish_id] = {}
                    if current_zone not in zone_detection_counter[fish_id]:
                        zone_detection_counter[fish_id][current_zone] = 0
                    zone_detection_counter[fish_id][current_zone] += 1
 
                    threshold = fps/2
                    if zone_detection_counter[fish_id][current_zone] >= threshold and current_zone != 0:
                        if (current_zone not in zones_written) :
                            fish_zones[fish_id] = current_zone
                            zones_written[current_zone] = fish_id
                            zones_file.write(f"Frame {(index_frame) / fps:.2f}s: Fish {fish_id} transitioned to Zone {current_zone}, {zone_detection_counter[fish_id][current_zone]} \n")
                            if current_zone==5:
                                limit =1
 
                except IndexError:
                    pass
 
        boxes_file.write(f"\n")
        index_frame += 1
        frame = cv.resize(frame, (880, 624))
        cv.imshow('Lines', frame)
        if cv.waitKey(1) & 0xFF == ord('q') or limit==1:
            break
 
video.release()
cv.destroyAllWindows()