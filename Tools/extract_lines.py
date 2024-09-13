#Extract delimitation lines

import numpy as np
import pandas as pd
import cv2 as cv

def line_angle(x1, y1, x2, y2):
    return np.degrees(np.arctan2(y2 - y1, x2 - x1))


def extract_lines(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    edges = cv.Canny(blurred, 50, 150, apertureSize=3)
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, threshold=150, minLineLength=50, maxLineGap=10)
    edges = cv.resize(edges, (880, 624))
    cv.imshow('Loaded Image', edges)
    cv.waitKey(0)  # Wait indefinitely for a key press
    cv.destroyAllWindows()
    return lines


def detect_line_areas(df, lines, x1_tank, y1_tank, x2_tank, y2_tank):
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = line_angle(x1, y1, x2, y2)

            # Tighten the angle threshold for more accurate horizontal line detection
            if -7 <= angle <= 7:
                if (x1 > x1_tank and x2 < x2_tank) and (y1 > y1_tank and y2 < y2_tank):
                    df = pd.concat([df, pd.DataFrame([{"x1": x1, "y1": y1, "x2": x2, "y2": y2}])], ignore_index=True)
    df_sorted = df.sort_values(by="y1")
    yy = 0
    # Get the first row
    first_row = df_sorted['y1'].iloc[0]
    # Get the last row
    last_row = df_sorted['y1'].iloc[-1]
    
#I calculate the difference between the first and the last line then I divide by the number of lines to have the mean difference
    j = (last_row -first_row)/4
    areas = pd.DataFrame(columns=['x1', 'y1', 'x2', 'y2'])
    for index, row in df_sorted.iterrows():
        if np.abs(row['y1'] - yy) > j :
            yy = row['y1']
            areas = pd.concat([areas, df_sorted[df_sorted['y1'] == yy]], ignore_index=True)
    areas = areas.sort_values(by="y1")
    return areas

frame = cv.imread('D:/Chris/LabirintoLubinas/frame_5000.jpg')
height, width, channels = frame.shape
x1_tank = width / 4
y1_tank = height / 12
x2_tank, y2_tank = width - x1_tank, height - y1_tank
lines = extract_lines(frame)
df = pd.DataFrame(columns=['x1', 'y1', 'x2', 'y2'])
areas = detect_line_areas(df, lines, x1_tank, y1_tank, x2_tank, y2_tank)

for index, area in areas.iterrows():    
        cv.line(frame, (area['x1'], area['y1']), (area['x2'], area['y2']), (0, 255, 0), 2)
        print(area['y1'])
        
print(len(areas))
frame = cv.resize(frame, (880, 624))
cv.imshow('Loaded Image', frame)
cv.waitKey(0)  # Wait indefinitely for a key press
cv.destroyAllWindows()
