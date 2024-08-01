import cv2 as cv
import numpy as np
import pandas as pd

import torch
from ultralytics import YOLO
import time

# Helper function to calculate the angle of a line
def line_angle(x1, y1, x2, y2):
    return np.degrees(np.arctan2(y2 - y1, x2 - x1))

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
            # Draw bounding box around detected blue cap
            cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv.putText(frame, 'Bottle', (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            # Store the position of the bounding box
            positions.append((x, y, w, h))

    return frame, positions

# Load the YOLO model and ensure it uses the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLO("D:/FutureExpertData/Computervision/best.pt").to(device)


# Open the video file
video = cv.VideoCapture("D:/FutureExpertData/Computervision/GX016362.MP4")
#video = cv.VideoCapture("L:/laberinto/GX016355.MP4")
#video = cv.VideoCapture("D:/Chris/Rocio/Lubinas/Laberinto/Laberinto_anotaciones/cortado_laberinto2.mp4")
#video = cv.VideoCapture("D:/Chris/Rocio/Lubinas/Laberinto/Laberinto_anotaciones/cortado_laberinto3.mp4")
#video = cv.VideoCapture("./cortado_laberinto3.mp4")


# Initialize dictionary to track fish zones
fish_zones = {}
initial_zones = {}


# Open a text file to write the results
i1, i2, i3, i4, i5= 0, 0, 0, 0, 0
index_frame = 0
xmin = 2000
xmax = 0
areas = []
frame_number=0
box_limit = 0
#dictionary of zone where we first saw a fish
zones_written = {}

# Initialize dictionary to track fish zones
fish_zones = {}

# Initialize dictionary to count consecutive frames in a zone
zone_detection_counter = {}


with open("zone_transitions.txt", "w") as zones_file,  \
     open("lines_positions.txt", "w") as lines_file, \
     open("boxes_positions.txt", "w") as boxes_file, \
     open("bottle_position.txt", "w") as bottle_file :

    while True:
        ret, frame = video.read()
        if not ret:
            break
        # Get the image height and width
        height, width, channels = frame.shape

        # Top-left corner is always (0, 0)
        top_left = (0, 0)

        # Bottom-right corner coordinates based on width and height
        bottom_right = (width - 1, height - 1)

        print("Top-left:", top_left)
        print("Bottom-right:", bottom_right)

    # Identify the parts of the video where the tank is supposed to be
        x1_tank = width/4
        y1_tank = height/12


        # Perform object detection with YOLO
        results = model.track(frame, persist=True)

        # Ensure results are in the expected format
        boxes = results[0].boxes  # Assuming results[0] contains the detected objects for the current frame

        # Convert the frame to grayscale
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise and improve edge detection
        blurred = cv.GaussianBlur(gray, (5, 5), 0)

        # Detect edges using Canny edge detector
        edges = cv.Canny(blurred, 50, 150, apertureSize=3)

        # Detect lines using Hough Line Transform
        lines = cv.HoughLinesP(edges, 1, np.pi / 180, threshold=150, minLineLength=50, maxLineGap=10)

     # Obtain the dimensions of the image
    ## Add in a file named lines_positions.txt  for each frame the position of the 4 coordinates of the lines ##
        lines_file.write(f"Frame {index_frame} \n")
        df = pd.DataFrame(columns=['x1', 'y1', 'x2', 'y2'])
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Calculate the angle of the line
                angle = line_angle(x1, y1, x2, y2)

                # Filter for horizontal lines
                if -10 <= angle <= 10 or 170 <= angle <= 190:  # Angle close to 0 or 180 degrees
                    x2_tank, y2_tank = width - x1_tank, height- y1_tank
                    if (x1 > x1_tank and x2 < x2_tank) and (y1>y1_tank and y2< y2_tank ) : # We conserve only the one that are situated in the middle (what correspond to the tank's ones)
                        df = df._append({"x1": x1, "y1": y1, "x2": x2, "y2": y2}, ignore_index=True)
                        #if frame_number <500:
                    
                            #cv.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            #print(y1)

        # Sort the DataFrame by the 'y1' column
        df_sorted = df.sort_values(by="y1")
        yy = 0
        index_line = 1
        areas = []
           

        # Loop through each row in the DataFrame to draw only a single line for each area
        for index, row in df_sorted.iterrows():
            #print(f"({row['x1'], {row['y1']}}) ({row['x2']}, {row['y2']})")
            # Check if the value in the 'y1' column is greater than 5
            if np.abs(row['y1'] - yy) > 100:
                yy = row['y1']
                lines_file.write(f"Line {index_line}, Coordinates : ({row['x1']}, {row['y1']}, {row['x2']}, {row['y2']})\n")
                areas.append(yy)
                index_line+=1
                if frame_number>=500:
                    cv.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        lines_file.write(f"\n")
        index_frame+=1


     # Detect the blue cap (bottle) using color segmentation
        
        frame, positions = detect_blue_cap(frame, x1_tank, width)

        # Print positions of detected blue caps
        #print(f"Position: x={position[0]}, y={position[1]}, width={position[2]+ position[0]}, height={position[3] + position[1]}")

        # Write in a text file all limitating lines from each frame
        for pos in positions:
            #print(f"Position: x={pos[0]}, y={pos[1]}, width={pos[2]}, height={pos[3]}")
            y_botella = pos[3] + pos[1]
            ###Add y_botella in a file named bottle_position.txt###
            if box_limit ==0:
                bottle_file.write(f"Position: x={pos[0]}, y={pos[1]}, width={pos[2]}, height={pos[3]}, y_botella={y_botella}\n")
                box_limit =1


        # Extract bounding box coordinates and store them
        frame_bboxes = []
        boxes_file.write(f"Frame {index_frame} \n")
        for idx, box in enumerate(boxes):
            if idx >= 5:  # Limit to 5 bounding boxes
                break
            # Extract coordinates (xmin, ymin, xmax, ymax) and confidence from the bounding box
            xmin, ymin, xmax, ymax = box.xyxy[0].tolist()  # Convert to list for easier manipulation
            confidence = box.conf[0].item()  # Extract the confidence score
            frame_bboxes.append((idx, xmin, ymin, xmax, ymax, confidence))  # Include the ID and confidence
            center_y = (ymin + ymax) / 2
            # print("posiciones de las lubinas")
            print(ymin, ymax)
            ### add in a file named boxes_positions.txt for each frame the position of each bounding box with its ID ##
            boxes_file.write(f"ID: {idx}, BBox: ({xmin}, {ymin}, {xmax}, {ymax}), Confidence: {confidence:.2f}\n")
            if confidence > 0.55:
                try:
                    # Determine the current zone of the fish based on its position
                    current_zone = 0

                    # Repartimos nombres a las zonas con respecto a la position de la potella (si está arriba o abajo)
                    if y_botella > height / 2:
                        if ymin > 0 and ymin <= areas[0]:
                            current_zone = 1
                        elif ymin > areas[0] + (areas[0]+areas[1])/2 and ymin <= areas[1] :
                            current_zone = 2
                        elif ymin > areas[1] + (areas[1]+areas[2])/2  and ymin <= areas[2]:
                            current_zone = 3
                        elif ymin > areas[2] + (areas[2]+areas[3])/2  and ymin <= areas[3]:
                            current_zone = 4
                        else:
                            current_zone = 5
                    else:
                        if ymax < height and ymax >= areas[len(areas) - 1]:
                            current_zone = 1
                        elif ymax < areas[len(areas) - 1] + (areas[len(areas) - 1] + areas[len(areas) - 2])/2 and ymax >= areas[len(areas) - 2]:
                            current_zone = 2
                        elif ymax < areas[len(areas) - 2] + (areas[len(areas) - 2] + areas[len(areas) - 3])/2 and ymax >= areas[len(areas) - 3]:
                            current_zone = 3
                        elif ymax < areas[len(areas) - 3] + (areas[len(areas) - 3] + areas[len(areas) - 4])/2 and ymax >= areas[len(areas) - 4]:
                            current_zone = 4
                        else:
                            current_zone = 5
                  # Initialize or update the detection counter for the fish in the current zone
                    fish_id = idx
                    if fish_id not in zone_detection_counter:
                        zone_detection_counter[fish_id] = {}
                    if current_zone not in zone_detection_counter[fish_id]:
                        zone_detection_counter[fish_id][current_zone] = 0
                    zone_detection_counter[fish_id][current_zone] += 1

                    # Check if the zone detection is consistent for more than 100 frames
                    if zone_detection_counter[fish_id][current_zone] >= 100:
                        if current_zone not in zones_written:
                            fish_zones[fish_id] = current_zone
                            zones_written[current_zone] = fish_id
                            zones_file.write(f"Frame {index_frame}: Fish {fish_id} transitioned to Zone {current_zone}\n")
                    
                    #cv.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
                    #cv.putText(frame, f'ID: {idx}, zone: {current_zone}', (int(xmin), int(ymin) - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                except IndexError:
                    pass

                        
           

    # Display the frame with detected edges and lines
        frame = cv.resize(frame,(1280,1024))
        cv.imshow('Lines', frame)

        # Wait for a key press to close the window
        if cv.waitKey(5) == ord('x'):
            break

video.release()
cv.destroyAllWindows()
