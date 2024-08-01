#this part include the head the body and not only the body




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
#video = cv.VideoCapture("D:/FutureExpertData/Computervision/GX016362.MP4")
video = cv.VideoCapture("D:/FutureExpertData/Computervision/GX016355.MP4")
#video = cv.VideoCapture("D:/FutureExpertData/Computervision/GX016450.MP4")
#video = cv.VideoCapture("D:/FutureExpertData/Computervision/chrisus.MP4")
 
fish_zones = {}
zone_detection_counter = {}
zones_written = {}

fish_zones2 = {}
zone_detection_counter2 = {}
zones_written2 = {}
 
fps = video.get(cv.CAP_PROP_FPS)
limit =0
time=0
current_zone=0
with open("zone_transitions.txt", "w") as zones_file, \
     open("lines_positions.txt", "w") as lines_file, \
     open("boxes_positions.txt", "w") as boxes_file, \
     open("bottle_position.txt", "w") as bottle_file:
   
    nframe = extract_frame(video, 500)
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
        
        
                #When only the head crosses the line
        #differents areas
        area0 = (areas['y1'].iloc[0] + areas['y2'].iloc[0])/2
        area1 = (areas['y1'].iloc[1] + areas['y2'].iloc[1])/2
        area2 = (areas['y1'].iloc[2] + areas['y2'].iloc[2])/2
        area3 = (areas['y1'].iloc[3] + areas['y2'].iloc[3])/2
        
        
        for idx, box in enumerate(boxes):
            if idx >= 5:
                break
            xmin, ymin, xmax, ymax = box.xyxy[0].tolist()
            confidence = box.conf[0].item()
            if confidence > 0.55:
                try:
                    
                    current_zone = 0
                    x_tank = (x2_tank + x1_tank) / 2
                    if y_botella > height / 2:
                        if len(areas) >= 4:
                            if ymax > 0 and ymax <=area0:
                                current_zone = 1
                            elif (ymax > area0 and ymax <= area1) and xmax < x_tank:
                                current_zone = 2
                            elif (ymax > area1 +10 and ymax <= area2) and xmax > x_tank:
                                current_zone = 3
                            elif (ymax > area2 and  ymax <=area3) and xmax < x_tank:
                                current_zone = 4
                            elif (ymax > area3) and xmax > x_tank:
                                current_zone = 5
                    else:
                        if len(areas) >= 4:
                            if ymin < height and ymin >= area3:
                                current_zone = 1
                            elif (ymin < area3 and ymin >= area2) and xmax > x_tank:
                                current_zone = 2
                            elif (ymin < area2 and ymin >= area1) and xmax < x_tank:
                                current_zone = 3
                            elif (ymin < area1 and ymin >= area0) and xmax > x_tank:
                                current_zone = 4
                            elif (ymin < area0+100) and xmax < x_tank:
                                current_zone = 5
                    #cv.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
                    #cv.putText(frame, f'ID: {idx}, zone: {current_zone}', (int(xmin), int(ymin) - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    fish_id = idx
                    if fish_id not in zone_detection_counter2:
                        zone_detection_counter2[fish_id] = {}
                    if current_zone not in zone_detection_counter2[fish_id]:
                        zone_detection_counter2[fish_id][current_zone] = 0
                    zone_detection_counter2[fish_id][current_zone] += 1
 
                    threshold = fps*0.75
                    time = (index_frame) / fps
                    if zone_detection_counter2[fish_id][current_zone] >= threshold and current_zone != 0:
                        #if (current_zone not in zones_written) and (current_zone == 1 or current_zone == list(zones_written.keys())[-1] + 1):
                        if (current_zone not in zones_written2) and (current_zone==1 or current_zone == list(zones_written2.keys())[-1]+1):
                            fish_zones2[fish_id] = current_zone
                            zones_written2[current_zone] = fish_id
                            zones_file.write(f"Cabeza\n")
                            zones_file.write(f"Tiempo {time:.2f}s: Lubina {fish_id} llegada a la zona {current_zone}\n")
                except IndexError:
                    pass
                    
        
        #When all the body crosses the line
        
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
                            if center_y > 0 and center_y <= area0 :
                                current_zone = 1
                            elif (center_y > area0 and center_y <= area1 ) and xmax < x_tank:
                                current_zone = 2
                            elif (center_y > area1 +10 and center_y <= area2) and xmax > x_tank:
                                current_zone = 3
                            elif (center_y > area2 and center_y <= area3) and xmax < x_tank:
                                current_zone = 4
                            elif (center_y > area3) and xmax > x_tank:
                                current_zone = 5
                    else:
                        if len(areas) >= 4:
                            if center_y < height and center_y >= area1:
                                current_zone = 1
                            elif (center_y < area1 and center_y >= area2) and xmax > x_tank:
                                current_zone = 2
                            elif (center_y < area2 and center_y >= area1) and xmax < x_tank:
                                current_zone = 3
                            elif (center_y < area1 and center_y >= area0) and xmax > x_tank:
                                current_zone = 4
                            elif (center_y < area0+100) and xmax < x_tank:
                                current_zone = 5
                    #cv.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
                    #cv.putText(frame, f'ID: {idx}, zone: {current_zone}', (int(xmin), int(ymin) - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    fish_id = idx
                    if fish_id not in zone_detection_counter:
                        zone_detection_counter[fish_id] = {}
                    if current_zone not in zone_detection_counter[fish_id]:
                        zone_detection_counter[fish_id][current_zone] = 0
                    zone_detection_counter[fish_id][current_zone] += 1
 
                    threshold = fps*0.75
                    time = (index_frame) / fps
                    if zone_detection_counter[fish_id][current_zone] >= threshold and current_zone != 0:
                        #if (current_zone not in zones_written) and (current_zone == 1 or current_zone == list(zones_written.keys())[-1] + 1):
                        if (current_zone not in zones_written) and (current_zone==1 or current_zone == list(zones_written.keys())[-1]+1):
                            fish_zones[fish_id] = current_zone
                            zones_written[current_zone] = fish_id
                            zones_file.write(f"Cuerpo\n")
                            zones_file.write(f"Tiempo {time:.2f}s: Lubina {fish_id} llegada a la zona {current_zone}\n")
 
                except IndexError:
                    pass
                        
        #When only the Tail crosses the line
        
        for idx, box in enumerate(boxes):
            if idx >= 5:
                break
            xmin, ymin, xmax, ymax = box.xyxy[0].tolist()
            confidence = box.conf[0].item()
            if confidence > 0.55:
                try:
                    
                    current_zone = 0
                    x_tank = (x2_tank + x1_tank) / 2
                    if y_botella > height / 2:
                        if len(areas) >= 4:
                            if ymin > 0 and ymin <= area0:
                                current_zone = 1
                            elif (ymin > area0 and ymin <= area1) and xmax < x_tank:
                                current_zone = 2
                            elif (ymin > area1 +10 and ymin <= area2) and xmax > x_tank:
                                current_zone = 3
                            elif (ymin > area2 and ymin <= area1) and xmax < x_tank:
                                current_zone = 4
                            elif (ymin > area1) and xmax > x_tank:
                                current_zone = 5
                    else:
                        if len(areas) >= 4:
                            if ymax < height and ymax >= area1:
                                current_zone = 1
                            elif (ymax < area1 and ymax >= area2) and xmax > x_tank:
                                current_zone = 2
                            elif (ymax < area2 and ymax >= area1) and xmax < x_tank:
                                current_zone = 3
                            elif (ymax < area1 and ymax >= area0) and xmax > x_tank:
                                current_zone = 4
                            elif (ymax < area0+100) and xmax < x_tank:
                                current_zone = 5
                
                    #cv.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
                    #cv.putText(frame, f'ID: {idx}, zone: {current_zone}', (int(xmin), int(ymin) - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    fish_id = idx
                    if fish_id not in zone_detection_counter2:
                        zone_detection_counter2[fish_id] = {}
                    if current_zone not in zone_detection_counter2[fish_id]:
                        zone_detection_counter2[fish_id][current_zone] = 0
                    zone_detection_counter2[fish_id][current_zone] += 1
 
                    threshold = fps*0.75
                    time = (index_frame) / fps
                    if zone_detection_counter2[fish_id][current_zone] >= threshold and current_zone != 0:
                        #if (current_zone not in zones_written) and (current_zone == 1 or current_zone == list(zones_written.keys())[-1] + 1):
                        if (current_zone not in zones_written2) and (current_zone==1 or current_zone == list(zones_written2.keys())[-1]+1):
                            fish_zones2[fish_id] = current_zone
                            zones_written2[current_zone] = fish_id
                            zones_file.write(f"Cola\n")
                            zones_file.write(f"Tiempo {time:.2f}s: Lubina {fish_id} llegada a la zona {current_zone}\n")
                            if current_zone==5:
                                limit=1
                except IndexError:
                    pass

        print(f"{time:.2f}s \n")
        boxes_file.write(f"\n")
        index_frame += 1
        frame = cv.resize(frame, (880, 624))
        cv.imshow('Lines', frame)
        if cv.waitKey(1) & 0xFF == ord('q') or limit==1:
            break
    
 
video.release()
cv.destroyAllWindows()