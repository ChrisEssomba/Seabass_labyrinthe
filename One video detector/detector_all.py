#6.2 + tracking
#Import the necessary libraries
import os
import cv2 as cv
import numpy as np
import pandas as pd
from ultralytics import YOLO
import gc
 
 
# The function that calculates the angle of a line based on this coordinates
def line_angle(x1, y1, x2, y2):
    return np.degrees(np.arctan2(y2 - y1, x2 - x1))
 
# The function that extract a certain frame from a video
def extract_frame(video, frame_number):
    if not video.isOpened():
        print("Error: Could not open video.")
        exit()
    video.set(cv.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = video.read()
    if ret:
        cv.imwrite('frame_2.jpg', frame)
    else:
        print(f"Error: Could not retrieve frame {frame_number}.")
    video.set(cv.CAP_PROP_POS_FRAMES, 0)
    return frame
 
#The function that extract the coordinates of a tank
def tank_coordinates(frame):
    height, width,_ = frame.shape
    x1_tank = width / 4
    y1_tank = height / 12
    x2_tank, y2_tank = width - x1_tank, height - y1_tank
 
    return x1_tank, y1_tank,x2_tank, y2_tank
 
# The function that extracts all the lines present in a frame
def extract_lines(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    edges = cv.Canny(blurred, 50, 150, apertureSize=3)
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, threshold=150, minLineLength=50, maxLineGap=10)
    return lines
 
# The function that extracts the delimation lines of a tank
def detect_line_areas(lines, x1_tank, y1_tank, x2_tank, y2_tank):
    df = pd.DataFrame(columns=['x1', 'y1', 'x2', 'y2'])
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
    delimitation_lines = pd.DataFrame(columns=['x1', 'y1', 'x2', 'y2'])
    for index, row in df_sorted.iterrows():
        if np.abs(row['y1'] - yy) > j:
            yy = row['y1']
            delimitation_lines = pd.concat([delimitation_lines, df_sorted[df_sorted['y1'] == yy]], ignore_index=True)
    delimitation_lines = delimitation_lines.sort_values(by="y1")
    return delimitation_lines
 
# The function that returns the position of the bottle of food
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
 
    return positions[0]


 
# Load the YOLO model and video
model = YOLO("D:/FutureExpertData/Computervision/best.pt")
 
# Open the video file
#video = cv.VideoCapture("D:/FutureExpertData/Computervision/GX016362.MP4")
#video = cv.VideoCapture("D:/FutureExpertData/Computervision/GX016355.MP4")
video = cv.VideoCapture("D:\FutureExpertData\Computervision\cortado_laberinto2.mp4")
#video = cv.VideoCapture("D:/FutureExpertData/Computervision/GX016450.MP4")
#video = cv.VideoCapture("F:\ENVIROBASS_NOE\TESTS\APRENDIZAJE\S_26_29_MAYO\S_26_29_D1\S_26_29_D1_B3\GX016349.MP4")
             
# Get the total number of frames in the video
total_frames = int(video.get(cv.CAP_PROP_FRAME_COUNT))

# Calculate the middle frame index
middle_frame_index = total_frames // 2 + 100

#Initialize the dictionary that'll contain the id and the zone of the first seabass that will this one
zones_written = {}
zones_written2 = {}
zones_written3 = {}
#Initialize the dictionary that'll contain the bounding box id and its current zone
dic_zones = {}
dic_zones2 = {}
dic_zones3 = {}
#Initialize the dictionary that'll contain the bounding box id and its occurence number in its current zone
dic_occ = {}
dic_occ2 = {}
dic_occ3 = {}

from collections import defaultdict

# Create a dictionary to store the center coordinates and associated IDs
center_to_id = defaultdict(int)

# Get the video fps number
fps = video.get(cv.CAP_PROP_FPS)

#Initiliaze the limit and the time
limit =0
time=0
current_zone=0
cool = False

frame_width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('output8.avi', fourcc, fps, (frame_width, frame_height))

with open("zone_transitions.txt", "w") as zones_file, \
     open("lines_positions.txt", "w") as lines_file, \
     open("boxes_positions.txt", "w") as boxes_file, \
     open("bottle_position.txt", "w") as bottle_file:
   
 
    while cool is False:       
        # Get the frame located at the middle of the video
        middle_frame = extract_frame(video, middle_frame_index)

        x1_tank, y1_tank, x2_tank, y2_tank= tank_coordinates(middle_frame)
        # Get the all lines presented in the middle frame
        lines = extract_lines(middle_frame)

        # Extract the delimitation lines and write their coordinates in a file
        delimitation_lines = detect_line_areas(lines, x1_tank, y1_tank, x2_tank, y2_tank)
        for index, line in delimitation_lines.iterrows():    
            lines_file.write(f"Line {index}, Coordinates : ({line['x1']}, {line['y1']}, {line['x2']}, {line['y2']})\n")
        if len(delimitation_lines) !=4:
            middle_frame_index+= 1
        else:
            cool=True        
    
        
    # Get the coordintates of the bottle from the middle frame
    
    _,width,_ = middle_frame.shape
    position = detect_blue_cap(middle_frame, x1_tank, width)
    x_bottle, y_bottle, width_bottle, height_bottle= position[0], position[1], position[2], position[3]  
    # Write down in a file these coordinates
    bottle_file.write(f"Position: x={x_bottle}, y={y_bottle}, width={width_bottle}, height={height_bottle} \n")



    # Initialize the variable that stores the frame's index
    index_frame = 0

    # Initialize the variable that stores the frame that came before the current one
    prev_frame = None

    # Launch the video lecture
    while True:
        ret, frame = video.read()
        #frame = np.uint8(frame)
        if not ret:
            break
        if frame is None:
            break

        # Get the frame's shape
        height, width, channels = frame.shape
        
        # Get the tank's coordinates
        x1_tank, y1_tank,x2_tank, y2_tank = tank_coordinates(frame)

        # check if there was a frame before the current one
        if prev_frame is not None and frame.shape != prev_frame.shape:
            # Ensure frame sizes are consistent
            if frame.shape != prev_frame.shape:
                print(f"Frame size mismatch: {frame.shape} vs {prev_frame.shape}")
                model = YOLO("D:/FutureExpertData/Computervision/best.pt")
                continue
    
        
        # Use the track() function to detect and track the seabasses in the current frame and then store the results
        try:
            results = model.track(frame, persist=True)
        except cv.error as e:
            print(f"OpenCV error occurred: {e}")
            continue

        # Use the results stores to extract the bounding boxes
        boxes = results[0].boxes

    
        # Draw bounding box around detected blue cap
        cv.rectangle(frame, (x_bottle, y_bottle), (x_bottle + width_bottle, y_bottle + height_bottle), (255, 0, 0), 2)
        cv.putText(frame, 'Bottle', (x_bottle, y_bottle - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)



        # Display the delimitating lines
        for index, line in delimitation_lines.iterrows():    
            cv.line(frame, (line['x1'], line['y1']), (line['x2'], line['y2']), (0, 255, 0), 2)

        # Write in a file the current index frame
        boxes_file.write(f"Frame {index_frame} \n")

        for idx, box in enumerate(boxes):
            # Ensure we don't have more than 5 bounding boxes
            if idx >= 5:
                break

            # Get the coordinate, confidence level and y center of each bounding box
            xmin, ymin, xmax, ymax = box.xyxy[0].tolist()
            confidence = box.conf[0].item()
            y_center = (ymin + ymax) / 2

            # Calculate the center of the bounding box
            center_x = (xmin + xmax) / 2
            center_y = y_center

            # Use the center coordinates as the unique identifier
            center_key = (center_x, center_y)

            # If the center is not yet associated with an ID, assign a new one
            if center_key not in center_to_id:
                center_to_id[center_key] = idx

            # Write down in a file the bounding box coordinates and associated ID
            boxes_file.write(f"ID: {center_to_id[center_key]}, BBox: ({xmin}, {ymin}, {xmax}, {ymax}), Confidence: {confidence:.2f}\n")
            
            # Given that the bottle is supposed to be located in the middle of the tank, we attribute its value to the corresponding variable
            x_middle_tank = x_bottle

            # Store each delimitation line in a separate variable
            line1 = delimitation_lines['y1'].iloc[0]
            line2 = delimitation_lines['y1'].iloc[1]
            line3 = delimitation_lines['y1'].iloc[2]
            line4 = delimitation_lines['y1'].iloc[3]
            
           
            # To reduce the false positive observations we work only with the bounding boxes having a confidence level greater than 0.55 
            # Detect the head of the fish
            if confidence > 0.55:       
                                
                # Display the bounding box
                cv.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
                cv.putText(frame, f'ID: {idx}, zone: {current_zone}', (int(xmin), int(ymin) - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                try:
                    # Initialize the variable that contains the zone where the bounding box is located in the current frame
                    current_zone = 0
                    #x_tank = (x2_tank + x1_tank) / 2

                    # Handle the case where the bottle of food is located at the bottom of the tank
                    if y_bottle > height / 2:      
                    # We follow the logic according which, a zone is attributed to a bounding box if its center is located between two consecutives lines and on the same side than the opening that gives to this zone
                    # With some specificties for the zone 1 and 5
                    # Because the endpoint (bottle of food) is at the bottom of the tank we start counting from the very first pixel (0)
                            if ymax > 0 and ymax <= line1:
                                current_zone = 1
                            elif (ymax > line1 and ymax <= line2) and xmax < x_middle_tank:
                                current_zone = 2
                            elif (ymax > line2 and ymax <= line3) and xmin > x_middle_tank:
                                current_zone = 3
                            elif (ymax > line3 and ymax <= line4) and xmax < x_middle_tank:
                                current_zone = 4
                            elif (ymax > line4) and xmin > x_middle_tank:
                                current_zone = 5
                            else:
                                current_zone=0

                    # Handle the case where the bottle of food is located at the top of the tank
                    else:
                    # Because the endpoint (bottle of food) is at the top of the tank we start counting from the very last pixel (height)
                            if ymin < height and ymin >= line4:
                                current_zone = 1
                            elif (ymin < line4 and ymin >= line3) and xmin > x_middle_tank:
                                current_zone = 2
                            elif (ymin < line3 and ymin >= line2) and xmax < x_middle_tank:
                                current_zone = 3
                            elif (ymin < line2 and ymin >= line1) and xmin > x_middle_tank:
                                current_zone = 4
                            elif (ymin < line1+100) and xmax < x_middle_tank:
                                current_zone = 5
                            else:
                                current_zone=0

                    # In a first dictionary (dic_zone) we associate each id with a zone and in a second dictionary (dic_occ) the same uid with its number of occurences in this zone
                    # Calculate the number of consecutive appearances of a bounding in a zone
                    if idx not in dic_zones3:
                        dic_zones3[idx] = current_zone
                        dic_occ3[idx]=1
                    else:
                        if dic_zones3[idx] == current_zone:
                            dic_occ3[idx] +=1
                        else:
                            dic_zones3[idx] = current_zone
                            dic_occ3[idx]=1

                    # Set a threshold
                    threshold = 10
                    if dic_occ3[idx] >= threshold and current_zone != 0:
                        if (current_zone not in zones_written3 and current_zone==1) or (current_zone not in zones_written3 and current_zone == list(zones_written3.keys())[-1]+1):
            
                            zones_written3[current_zone] = idx
                            zones_file.write(f"La Cabeza de la lubina {idx} ha passado por la zona {current_zone} en {time:.2f}s:, {dic_occ3[idx]} \n")  

                except IndexError:
                    pass
        
            # Detect the body of the fish
            if confidence > 0.55:

                try:
                    current_zone = 0
                    if y_bottle > height / 2:      
                            if y_center > 0 and y_center <= line1:
                                current_zone = 1
                            elif (y_center > line1 and y_center <= line2) and xmax < x_middle_tank:
                                current_zone = 2
                            elif (y_center > line2 and y_center <= line3) and xmin > x_middle_tank:
                                current_zone = 3
                            elif (y_center > line3 and y_center <= line4) and xmax < x_middle_tank:
                                current_zone = 4
                            elif (y_center > line4) and xmin > x_middle_tank:
                                current_zone = 5
                            else:
                                current_zone=0
                    else:
                            if y_center < height and y_center >= line4:
                                current_zone = 1
                            elif (y_center < line4 and y_center >= line3) and xmin > x_middle_tank:
                                current_zone = 2
                            elif (y_center < line3 and y_center >= line2) and xmax < x_middle_tank:
                                current_zone = 3
                            elif (y_center < line2 and y_center >= line1) and xmin > x_middle_tank:
                                current_zone = 4
                            elif (y_center < line1+100) and xmax < x_middle_tank:
                                current_zone = 5
                            else:
                                current_zone=0

                    # In a first dictionary (dic_zone) we associate each id with a zone and in a second dictionary (dic_occ) the same uid with its number of occurences in this zone
                    # Calculate the number of consecutive appearances of a bounding in a zone
                    if idx not in dic_zones:
                        dic_zones[idx] = current_zone
                        dic_occ[idx]=1
                    else:
                        if dic_zones[idx] == current_zone:
                            dic_occ[idx] +=1
                        else:
                            dic_zones[idx] = current_zone
                            dic_occ[idx]=1

                    # Set a threshold
                    threshold = 10

                    # Calculate the current time in second
                    time = (index_frame) / fps

                    # Verify if the occurence of this bounding is greater or equal to the threshold
                    if dic_occ[idx] >= threshold and current_zone != 0:
                        #if (current_zone not in zones_written) and (current_zone == 1 or current_zone == list(zones_written.keys())[-1] + 1):
                        # Ensure that the zone written are consecutives
                        if (current_zone not in zones_written and current_zone==1) or (current_zone not in zones_written and current_zone == list(zones_written.keys())[-1]+1):
                            #fish_zones[fish_id] = current_zone
                            zones_written[current_zone] = idx
                            zones_file.write(f"El Cuerpo de la Lubina {idx} ha passado por la zona {current_zone} en {time:.2f}s:, {dic_occ[idx]} \n")
                            # Set a limit at 5, which means that when the first seabass crosses this zone the program should stops the processing of the this video and goes to the next
                            #if current_zone==5:
                             #   limit =1
                

                except IndexError:
                    pass
                
            # Detect the tail of the fish
            if confidence > 0.55:
                try:

                    current_zone = 0
             
                    if y_bottle > height / 2:      

                            if ymin > 0 and ymax <= line1:
                                current_zone = 1
                            elif (ymin > line1 and ymax <= line2) and xmax < x_middle_tank:
                                current_zone = 2
                            elif (ymin > line2 and ymax <= line3) and xmin > x_middle_tank:
                                current_zone = 3
                            elif (ymin > line3 and ymax <= line4) and xmax < x_middle_tank:
                                current_zone = 4
                            elif (ymin > line4) and xmin > x_middle_tank:
                                current_zone = 5
                            else:
                                current_zone=0

    
                    else:

                            if ymax < height and ymax >= line4:
                                current_zone = 1
                            elif (ymax < line4 and ymax >= line3) and xmin > x_middle_tank:
                                current_zone = 2
                            elif (ymax < line3 and ymax >= line2) and xmax < x_middle_tank:
                                current_zone = 3
                            elif (ymax < line2 and ymax >= line1) and xmin > x_middle_tank:
                                current_zone = 4
                            elif (ymax < line1+100) and xmax < x_middle_tank:
                                current_zone = 5
                            else:
                                current_zone=0
                    if idx not in dic_zones2:
                        dic_zones2[idx] = current_zone
                        dic_occ2[idx]=1
                    else:
                        if dic_zones2[idx] == current_zone:
                            dic_occ2[idx] +=1
                        else:
                            dic_zones2[idx] = current_zone
                            dic_occ2[idx]=1

                    # Set a threshold
                    threshold = 10
                    if dic_occ2[idx] >= threshold and current_zone != 0:
                        if (current_zone not in zones_written2 and current_zone==1) or (current_zone not in zones_written2 and current_zone == list(zones_written2.keys())[-1]+1):
                            zones_written2[current_zone] = idx
                            zones_file.write(f"La Cola de la lubina {idx} ha passado por la zona {current_zone} en {time:.2f}s:, {dic_occ2[idx]} \n")
                            if current_zone==5:
                               limit =1
                

                except IndexError:
                    pass
                # To reduce the false positive observations we work only with the bounding boxes having a confidence level greater than 0.55

           
        # Copy of the current frame before going to the next
        prev_frame = frame.copy()    

        #Display time
        print(f"{time:.2f}s \n")

        # Write the frame with the detected objects
        out.write(frame)  

        #Add space to bounding boxes file
        boxes_file.write(f"\n")

        #Incremente the frame index
        index_frame += 1

        #Display the frame
        frame = cv.resize(frame, (880, 624))
        cv.imshow('Lines', frame)
        
        
        # Call the garbage collector to cleanup the unused object in the memory after each video
        gc.collect()
        #Stop the process when the conditions are respected
        if cv.waitKey(1) & 0xFF == ord('q') or limit==1:
            break


video.release()
out.release()
cv.destroyAllWindows()
