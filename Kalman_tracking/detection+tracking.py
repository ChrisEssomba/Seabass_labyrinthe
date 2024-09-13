#Import the detecting libraries
import gc
import cv2 as cv
import numpy as np
import pandas as pd
from ultralytics import YOLO

# import tracking libraries
from gettext import install
from itertools import islice
import math
from random import randint
import cv2
from matplotlib import pyplot as plt
from filterpy.kalman import KalmanFilter as kf


 
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
        cv.imwrite('frame_5000èyh.jpg', frame)
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
    
    # Check if df_sorted is empty before proceeding
    if df.empty:
        print("No valid lines found within the tank area.")
        return pd.DataFrame(columns=['x1', 'y1', 'x2', 'y2'])

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
    if not positions:
        return None
        
 
    return positions[0]


# Definir la función that returns the bounding boxes and their confidence level
def detect_objects(image):
    outputs = model(image)
    boxes = outputs[0].boxes
    dict_bb = {}
    scores = {}
 
    for idx, box in enumerate(boxes):
        if idx >= 5:
            break
        dict_bb[idx] = box.xyxy[0].tolist()  # Mapping for each bounding box coordinate to an index
        scores[idx] = box.conf[0].item()
 
    return dict_bb, scores


# Load the YOLO model and video
model = YOLO("D:/Chris/LabirintoLubinas/best.pt")
 
# Open the video file
#video = cv.VideoCapture("D:/FutureExpertData/Computervision/GX016362.MP4")
#video = cv.VideoCapture("D:/FutureExpertData/Computervision/GX016355.MP4")
#video = cv.VideoCapture("D:/FutureExpertData/Computervision/GX016450.MP4")
#video = cv.VideoCapture("L:/ENVIROBASS_NOE/TESTS/APRENDIZAJE/S_26_29_MAYO/S_26_29_D1/S_26_29_D1_B1/GX036343.MP4")
video = cv.VideoCapture("D:/Chris/2023_PFG_Daniel_Areñas_Mayoral/defis.MP4")

# Get the total number of frames in the video
total_frames = int(video.get(cv.CAP_PROP_FRAME_COUNT))

# Calculate the middle frame index
middle_frame_index = total_frames // 2 + 100
   
#Initialize the dictionary that'll contain the id and the zone of the first seabass that will this one
zones_written = {}

#Initialize the dictionary that'll contain the bounding box id and its current zone
dic_zones = {}

#Initialize the dictionary that'll contain the bounding box id and its occurence number in its current zone
dic_occ = {}

# Get the video fps number 
fps = video.get(cv.CAP_PROP_FPS)
#frame_number = int(video.get(cv.CAP_PROP_POS_FRAMES))

# Get the bounding box colors
colors = [
    (255, 0, 0),   # Blue
    (255, 255, 255), # White
    (0, 0, 0),     # Black
    (0, 0, 255),   # Red
    (0, 255, 0)    # Green
]

# Inicializa el filtro del kalman
kalman_list = []
medidas = []
col_caja = []
mostrar = []
DictX = {}
DictY = {}
id = []
borrar = []
anteriores = []
dic_key = {}
dt = 1 / 15  # La cámara empleada es de 15 fps
petit_joueur=0
prev_box = {}
provision = 0
posicion_key ={}
occurence = {0:0,
         1:0,
         2:0,
         3:0,
         4:0}
breaks = 0

# Genero la resolución del video de salida
res = (int(1920), int(1080))  

# Configurar el video de salida
out = cv2.VideoWriter('D:/Chris/2023_PFG_Daniel_Areñas_Mayoral/Multi_kalman2.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, res)
 

#Initiliaze the limit and the time
limit =0
time=0
current_zone=0
cool = False
stop =0
with open("zone_transitions.txt", "w") as zones_file, \
     open("lines_positions.txt", "w") as lines_file, \
     open("boxes_positions.txt", "w") as boxes_file, \
     open("bottle_position.txt", "w") as bottle_file:
    while cool is False:      
        # Get the frame located at the middle of the video
        middle_frame = extract_frame(video, middle_frame_index)
        _,width,_ = middle_frame.shape
        
        #Get tank coordinates
        x1_tank, y1_tank, x2_tank, y2_tank= tank_coordinates(middle_frame)

        # Get the all lines presented in the middle frame
        lines = extract_lines(middle_frame)

        # Extract the delimitation lines and write their coordinates in a file
        delimitation_lines = detect_line_areas(lines, x1_tank, y1_tank, x2_tank, y2_tank)

        # Get the coordintates of the bottle from the middle frame
        position = detect_blue_cap(middle_frame, x1_tank, width)
        
        
        if len(delimitation_lines) !=4 or not position :
            middle_frame_index+= 1
        else:
            #write the lines positions
            for index, line in delimitation_lines.iterrows():    
                lines_file.write(f"Line {index}, Coordinates : ({line['x1']}, {line['y1']}, {line['x2']}, {line['y2']})\n")

            #write the bottle position
            x_bottle, y_bottle, width_bottle, height_bottle= position[0], position[1], position[2], position[3] 

            cool=True
        if middle_frame_index >= total_frames:
            print("Bottle not detected")
            break
        
    if cool:
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

        bob = {}
        bob2 = []
        frame = cv2.resize(frame, (1920,1080))
        boxes, scores = detect_objects(frame)
        a=0
        b=0

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
    
        '''
        # Use the track() function to detect and track the seabasses in the current frame and then store the results
        try:
            results = model.track(frame, persist=True)
        except cv.error as e:
            print(f"OpenCV error occurred: {e}")
            continue

        # Use the results stores to extract the bounding boxes
        boxes = results[0].boxes

        '''

    
        # Display the delimitating lines
        for index, line in delimitation_lines.iterrows():    
            cv.line(frame, (line['x1'], line['y1']), (line['x2'], line['y2']), (0, 255, 0), 2)

        # Write in a file the current index frame
        boxes_file.write(f"Frame {index_frame} \n")

        #We get the bounding boxes and their confidence levels
        boxes, scores = detect_objects(frame)
        score = scores.copy()

        # Get the first 5 items
        boxes = dict(islice(boxes.items(), 5))
        score = dict(islice(score.items(), 5))
        confidences = {}

        #We ensure that we deal only with the detections with a confidence level greater than 0.55
        if len(boxes)>=1:
            for key, value in scores.items():
                if value<=0.56:
                    boxes.pop(key)
                    score.pop(key)

           

        if len(boxes) <1:
              #We don't write
            out.write(frame)
            print("No fish detected in this frame.")
            index_frame+=1
            continue  # Skip the rest of the loop

    
        # Draw bounding box around detected blue cap
        cv.rectangle(frame, (x_bottle, y_bottle), (x_bottle + width_bottle, y_bottle + height_bottle), (255, 0, 0), 2)
        cv.putText(frame, 'Bottle', (x_bottle, y_bottle - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        #We initialize the kalman filter
        if len(boxes) > provision and len(boxes)<=5:
            for m in range(provision, len(boxes)):
                x0, y0, X0, Y0 = boxes[m]  # Accessing bounding box directly
                col_caja.append(colors[m])
                mostrar.append(colors[m])
                id.append(m)
                x = math.ceil(x0)
                y = math.ceil(y0)
                w0 = math.ceil(X0 - x0)
                h0 = math.ceil(Y0 - y0)
                X = np.array([x + w0 / 2])
                Y = np.array([y + h0 / 2])
                DictX[colors[m]] = X
                DictY[colors[m]] = Y
    
                # Generar filtro de kalman para ese objeto
                kalman = kf(dim_x=4, dim_z=2)
                kalman_list.append(kalman)
                kalman_list[-1].F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
                kalman_list[-1].H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
                kalman_list[-1].P *= 10
                kalman_list[-1].R = np.diag([1, 1])
                kalman_list[-1].Q = np.eye(4) * 5
    
                medidas.append(np.array([math.ceil(x + w0 / 2), math.ceil(y + h0 / 2)]).T)
                kalman_list[-1].x = np.array([medidas[-1][0], medidas[-1][1], 0, 0]).T
                dic_key[m] = 0
            #print("oulalalaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
            provision  = len(boxes)
 



        #Tracking + detection
        for j in range(len(kalman_list)):
        #*******************************************TRACKING LOGIC*********************************************************************
            cool = False
            distances = np.zeros(len(boxes))
            obj = np.zeros((len(boxes), 2))
    
            for key, value in boxes.items():
                obj[key] = [math.ceil(value[0] + (value[2] - value[0]) / 2), math.ceil(value[1] + (value[3] - value[1]) / 2)]
                distances[key] = math.dist(medidas[j], obj[key])

            min_index = np.argmin(distances)
            posicion = np.array([math.ceil(boxes[min_index][0] + (boxes[min_index][2] - boxes[min_index][0]) / 2),
                                math.ceil(boxes[min_index][1] + (boxes[min_index][3] - boxes[min_index][1]) / 2)])

            kalman_list[j].predict()
        
        ##Ici on gere le cas ou une bb disparait puis reapparait. Sachant que quand cela occur, le bb apparait avec un nouveau ID, different des precedents
            #Si il a perdu de vue sa lubina (donc elle a disparu) on compte l'occurence
            if distances[min_index] > 40:
                    posicion = anteriores[j]
                    dic_key[j]+=1
            #Si non on s'assure que son occurence reste 0
            else:
                dic_key[j] = 0

            #Dans le cas ou il se soit eloigné pendant une demi seconde (fps/2)
            if j in dic_key and dic_key[j]>=fps/2:
                #Si sa lubinas a reapparu cet a dire on est passé de 4 lubinas à 5 lubinas
                if len(boxes) == len(prev_box)+1:
                    for key, value in boxes.items():
                        #On cible l'id qui n'etait pas là au frame precedent et on actualise les coord de sa position avec ceux de la bb correspondante
                        if key not in prev_box:
                            posicion = np.array([math.ceil(value[0] + (value[2]- value[0] )/2),math.ceil(value[1] + (value[3]-value[1])/2)])
                            cool = True
                prev_box =boxes #On actualise le prev box seulement quand on est sur que ca da disparu pour que ca soit plus precis


            medidas[j] = posicion.reshape((2, 1))

            kalman_list[j].update(medidas[j])

            prediction = kalman_list[j].x
            x = prediction[0]
            y = prediction[1]

            center = (int(x), int(y))
            W = boxes[min_index][2] - boxes[min_index][0]
            H = boxes[min_index][3] - boxes[min_index][1]
            pt1 = (int(x - W / 2), int(y - H / 2))
            pt2 = (int(x + W / 2), int(y + H / 2))


        
            
            
        # print(posicion)
            
        
            # We store the bb associated to each kalman box and in the case where we have a kb that has lost it bb for at least 2sec we get it
            #Si une box de kal a perdu sa bb pendant plus de 2 sec et celles ci n'a pas disapparu (et reapparu)
            #Ici on gere le cas la box de kal a perdu sa bb sans que celle ci ne disparaisse peut etre a cause d'un mouvement brusque
            if dic_key[j]>=fps*2 and cool== False:
                b = 1
                petit_joueur2 = j
            else:
                #je stocke tous les id de bb qui sont associés à des kal
                for key, value in boxes.items():
                    if math.dist( (int(value[0]), int(value[1])), pt1) <=5:
                        confidences[j] = score[key]
                        bob2.append(key)
        
            #We handle this case making sure that we are at the last bb
            # Because we will replace it sociate bb with the that's not associated to anything else     
            #Dans le cas ou on a noté une box de kal perdu pendant au moins deux sec et que nous soyons en face du dernier element de la liste
            # cet adire on est sur que les id qui ne sont pas dans box n'ont tout simplement pas respecté la regle 
            if b==1 and j==len(kalman_list)-1:
                for key, value in boxes.items():
                    #On actualise cette box avec les coord de la bb qui n'est associé à aucune box kal
                    if key not in bob2:
                        pos2 = np.array([math.ceil(value[0] + (value[2]- value[0] )/2),math.ceil(value[1] + (value[3]-value[1])/2)])
                        medidas[petit_joueur2] = pos2
                        kalman_list[petit_joueur2].update(medidas[petit_joueur2])
                        b=0
                        #print('ggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg')
        
        
        #Ici j'ajoute les id de bb qui sont associé à une b de kal dans une variable
            for key, value in boxes.items():
                if math.dist( (int(value[0]), int(value[1])), pt1) <=5:
                    if key not in bob.values():
                        bob[j] = key
                    #Dans le cas ou une seconde boxe de kal se retrouve associé à la meme bb, je compte l'occurence
                    else:
                        #print("toooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo")
                        a=1
                        petit_joueur = j
                        pos_petit = pt1
                        occurence[petit_joueur]+=1
                        bob[j] = key

            #print(f" boxes = {len(boxes)} -> kalman = {len(kalman_list)}")

            

            #Si verifie s'il s'agit bien du cas ou deux boxes de kal suivent la meme lubina et je m'assure aussi que nous somme de l'indice de la derniere box de kal
            #Comme ca on est sur que les bb not prises en compte sont celles qui ne correspondent pas au critere
            if a==1 and j==len(kalman_list)-1 and occurence[petit_joueur]>=fps/2:
            
                # Lorsque deux lubinas se croisent les boxes de kal ont tendance à suivre une seule lubina et l'autre se retrouve sans rien
                #Dans ce cas je recupere les id de ces deux lubinas sous les noms de first and second key
                first_occurrence = {}
                first_key = None
                second_key = None
                for kb, bb in bob.items():
                    if bb in first_occurrence:
                        first_key = first_occurrence[bb]  # Get the first key where this value was seen
                        second_key = kb
                        break
                    else:
                        first_occurrence[bb] = kb  # Store the first key where the value appears


                for key, value in boxes.items():
                    #Je parcous les bb et je note celle qui n'est plus suivie
                    if key not in bob2 :
                        #A ce niveau on a deux boxes de kal qui suivent la meme lubinas, pour attribuer la bonne à chacune
                        #On verifie laquelles de la positions de ces boxes de kal avant leur choque est la plus proche de la position de bb non non tracké
                        #Celle dont la position avant la collision est la plus proche de la position actuel de la bb non tracké sera attribué à celle ci
                        if math.dist( (int(value[0]), int(value[1])), posicion_key[first_key]) < math.dist( (int(value[0]), int(value[1])), posicion_key[second_key]):
                            pos = np.array([math.ceil(value[0] + (value[2]- value[0] )/2),math.ceil(value[1] + (value[3]-value[1])/2)])
                            medidas[first_key] = pos
                            kalman_list[first_key].update(medidas[first_key])
                            #print('ggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg')
                            a=0
                            occurence[petit_joueur]=0
                        else:
                            pos = np.array([math.ceil(value[0] + (value[2]- value[0] )/2),math.ceil(value[1] + (value[3]-value[1])/2)])
                            medidas[second_key] = pos
                            kalman_list[second_key].update(medidas[second_key])
                            #print('ggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg')
                            a=0
                            occurence[petit_joueur]=0

            #We handle the case where there's more kalman boxes than bounding boxes in the tank

            if  j==len(kalman_list)-1 and occurence[petit_joueur]>=fps*2 and len(kalman_list)<4:
                k=0
                for key, value in boxes.items():
                    if key not in bob2 :
                        k=1
                if k==0:
                    occurence[petit_joueur] =0
                    provision -=1
                    # Remove Kalman filter at occurence 7
                    del kalman_list[petit_joueur]
                    # Adjust other lists to maintain alignment with kalman_list
                    del medidas[petit_joueur]
                    del col_caja[petit_joueur]
                    #del DictX[col_caja[petit_joueur]]
                    #del DictY[col_caja[petit_joueur]]
                    continue  # Skip the rest of the loop for j == 7


                
            #On associe a chaque id sa position, il est tout a la fin pour qu'il contiene la position du frame anterieur, et en cas de collision(a=1) on ne l'actualise pas
            if a==0:
                posicion_key[j] = pt1
        #***********************************************************DETECTING LOGIC*******************************************
   
            #Get the coordinate, confidence level and y center of each bouding box
            #xmin, ymin, xmax, ymax = box.xyxy[0].tolist()
            xmin, ymin = pt1
            xmax, ymax = pt2
            #confidence = confidences[j]
            y_center = (ymin + ymax) / 2
            # Write down in a file the bounding box coordinates
            boxes_file.write(f"ID: {j}, BBox: ({xmin}, {ymin}, {xmax}, {ymax})\n")

            # Given that the bottle is supposed to be located in the middle of the tank, we attribute its value to the corresponding variable
            x_middle_tank = x_bottle

            # Store each delimitation line in a separate variable
            line1 = delimitation_lines['y1'].iloc[0]
            line2 = delimitation_lines['y1'].iloc[1]
            line3 = delimitation_lines['y1'].iloc[2]
            line4 = delimitation_lines['y1'].iloc[3]
        
            # To reduce the false positive observations we work only with the bounding boxes having a confidence level greater than 0.55
            #if confidence > 0.55:
            try:
                # Initialize the variable that contains the zone where the bounding box is located in the current frame
                current_zone = 0
                #x_tank = (x2_tank + x1_tank) / 2

                # Handle the case where the bottle of food is located at the bottom of the tank
                if y_bottle > height / 2:      
                # We follow the logic according which, a zone is attributed to a bounding box if its center is located between two consecutives lines and on the same side than the opening that gives to this zone
                # With some specificties for the zone 1 and 5
                # Because the endpoint (bottle of food) is at the bottom of the tank we start counting from the very first pixel (0)
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

                # Handle the case where the bottle of food is located at the top of the tank
                else:
                # Because the endpoint (bottle of food) is at the top of the tank we start counting from the very last pixel (height)
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
                if j not in dic_zones:
                    dic_zones[j] = current_zone
                    dic_occ[j]=1
                else:
                    if dic_zones[j] == current_zone:
                        dic_occ[j] +=1
                    else:
                        dic_zones[j] = current_zone
                        dic_occ[j]=1

                # Set a threshold
                threshold = 10

                # Calculate the current time in second
                time = (index_frame) / fps

                # Verify if the occurence of this bounding is greater or equal to the threshold
                if dic_occ[j] >= threshold and current_zone != 0:
                    #if (current_zone not in zones_written) and (current_zone == 1 or current_zone == list(zones_written.keys())[-1] + 1):
                    # Ensure that the zone written are consecutives
                    if (current_zone not in zones_written and current_zone==1) or (current_zone not in zones_written and current_zone == list(zones_written.keys())[-1]+1):
                        #fish_zones[fish_id] = current_zone
                        zones_written[current_zone] = j
                        zones_file.write(f"Lubina {j} ha passado por la zona {current_zone} en {time:.2f}s:, {dic_occ[j]} \n")
                        # Set a limit at 5, which means that when the first seabass crosses this zone the program should stops the processing of the this video and goes to the next
                        if current_zone==5:
                            limit =1
            except IndexError:
                    pass
            
            #Drawing
            cv2.rectangle(frame, pt1, pt2, col_caja[j], 2, 1)
            cv2.putText(frame, f"{j}K", (int(x - W / 2), int(y - H / 2) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            #for key, value in boxes.items():
            #   if key == j:
            #      cv2.putText(frame, f"{key}B", (int(value[0]) +20, int(value[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            X = np.vstack((DictX[col_caja[j]], [x]))
            Y = np.vstack((DictY[col_caja[j]], [y]))
            DictX[col_caja[j]] = X
            DictY[col_caja[j]] = Y

            if distances[min_index] > 40:
                cv2.circle(frame, (int(posicion[0]), int(posicion[1])), 10,(0, 0, 255) , 3)
            else:
                cv2.circle(frame, (int(posicion[0]), int(posicion[1])), 10,  (255, 0, 0), 3)
            # Draw the trajectory line
            for i in range(1, len(X)):
                cv2.line(frame, (int(X[i-1]), int(Y[i-1])), (int(X[i]), int(Y[i])), col_caja[j], 1)

                
        anteriores = medidas

        #We don't write
        out.write(frame)

        # Copy of the current frame before going to the next
        prev_frame = frame.copy()    

        #Display time
        print(f"{time:.2f}s \n")

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

cv2.destroyAllWindows()
 