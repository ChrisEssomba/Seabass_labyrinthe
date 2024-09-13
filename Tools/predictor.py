#Simple script to predict a video

import cv2 as cv
from ultralytics import YOLO

# Load the YOLO model and ensure it uses the GPU
model = YOLO("D:/FutureExpertData/Computervision/best.pt")


# Open the video file
video = cv.VideoCapture("D:/FutureExpertData/Computervision/GX016362.mp4")

# Get video properties
fps = video.get(cv.CAP_PROP_FPS)
frame_width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('output2.avi', fourcc, fps, (frame_width, frame_height))
limit =0
while True:
    ret, frame = video.read()
    if not ret:
        break
    
    # Perform object detection with YOLO
    results = model(frame)
    
    # Draw bounding boxes and labels on the frame
    for result in results:
        boxes = result.boxes
        for box in boxes:
            xmin, ymin, xmax, ymax = map(int, box.xyxy[0].tolist())
            confidence = box.conf[0].item()
            label = box.cls[0].item()
            
            # Draw bounding box
            cv.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            
            # Draw label and confidence
            cv.putText(frame, f'{label} {confidence:.2f}', (xmin, ymin - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Write the frame with the detected objects
    out.write(frame)

    # Display the frame with bounding boxes
    cv.imshow('YOLO Prediction', frame)
    limit+=1
    if cv.waitKey(1) & 0xFF == ord('q') or limit==fps*300:
        break

# Release the video and writer, and close all windows
video.release()
out.release()
cv.destroyAllWindows()
