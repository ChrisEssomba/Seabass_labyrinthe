import re
import cv2 as cv
import os

# Paths
video_path = r"L:\ENVIROBASS_NOE\TESTS\APRENDIZAJE\S_9_12_JUNIO\S_9_12_D1\S_9_12_D1_B2\GX036447.MP4"
boxes_positions_path = r"D:\Chris\AllCodes\S_9_12_JUNIO\S_9_12_D1\S_9_12_D1_B2\GX036447_boundingBoxes.txt"
lines_path = r"D:\Chris\AllCodes\S_9_12_JUNIO\S_9_12_D1\S_9_12_D1_B2\GX036447_posLineas.txt"
# Check if video file exists
if not os.path.isfile(video_path):
    print(f"Error: Video file does not exist at {video_path}")
    exit()

# Check if bounding boxes file exists
if not os.path.isfile(boxes_positions_path):
    print(f"Error: Boxes positions file does not exist at {boxes_positions_path}")
    exit()

# Open the video
video = cv.VideoCapture(video_path)
video.set(cv.CAP_PROP_POS_FRAMES, 0)
if not video.isOpened():
    print(f"Error: Could not open video file {video_path}")
    exit()

# Read bounding boxes data
with open(boxes_positions_path, "r") as boxes_file:
    boxes_data = boxes_file.readlines()

# Create a window for displaying the video
cv.namedWindow("Video", cv.WINDOW_NORMAL)

frame_number = 0 
num_frames = int(video.get(cv.CAP_PROP_FRAME_COUNT))
data_index = 0

# Get the video fps number 
fps = video.get(cv.CAP_PROP_FPS)

# Read line coordinates from file
def read_lines(file_path):
    lines = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                match = re.search(r'Coordinates : \(([^)]+)\)', line)
                if match:
                    coords = tuple(map(int, match.group(1).split(', ')))
                    lines.append(coords)
    return lines
# Read line coordinates
line_coordinates = read_lines(lines_path)
while True:
    ret, frame = video.read()
    if not ret:
        break

    # Process bounding boxes for the current frame
    while data_index < len(boxes_data):
        boxes_line = boxes_data[data_index].strip()
        
        if boxes_line.startswith("Frame"):
            # If a new frame is detected, check if it's the correct frame
            if int(boxes_line.split()[1]) == frame_number:
                data_index += 1
                continue
            else:
                # Skip lines until the correct frame is found
                data_index += 1
                continue

        # If a blank line is encountered, break out of processing
        if not boxes_line:
            data_index += 1
            break
        
        # Extract coordinates from each bounding box line
        boxes = boxes_line.split("\n")
        for box in boxes:
            if box:
                coords = [val.split("=")[-1] for val in box.split(",")]
                if len(coords) == 4:
                    x_min, y_min, x_max, y_max = coords
                    # Convert coordinates to float
                    x_min, y_min, x_max, y_max = map(float, (x_min, y_min, x_max, y_max))
                    # Draw bounding box on the frame (convert to int for drawing)
                    cv.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
                    cv.putText(frame, f"Frame number : {frame_number}", (100,100), cv.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 10)
                    cv.putText(frame, f"Time : {frame_number/fps:.2f}s ", (100,200), cv.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 10)
                    #cv.putText(frame, f"{kalman_id}K", (int(x - W / 2), int(y - H / 2) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        data_index += 1
    # Draw lines on the frame
    for (x1, y1, x2, y2) in line_coordinates:
        cv.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Display the frame with bounding boxes
    cv.imshow("Video", frame)

    # Break loop on 'q' key press or if end of video
    if cv.waitKey(30) & 0xFF == ord('q'):
        break

    frame_number += 1
    if frame_number >= num_frames:
        break

# Release video and close windows
video.release()
cv.destroyAllWindows()
