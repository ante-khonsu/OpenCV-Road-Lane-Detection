import cv2
import numpy as np
from collections import deque


#change videos in line 120





# remembering last lines
left_history = deque(maxlen=5)
right_history = deque(maxlen=5)


def make_line_points(y1, y2, line):
    slope, intercept = line
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])


def average_slope_intercept(lines, height, width):
    left_lines = []
    right_lines = []

    if lines is None:
        return None, None

    for line in lines:
        x1, y1, x2, y2 = line[0]

        if x2 - x1 == 0:
            continue

        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1

        mid_x = width / 2

        #middle of the line
        x_mid = (x1 + x2) / 2
        y_mid = (y1 + y2) /2



        #filters
        length=np.sqrt((x2-x1)**2+(y2-y1)**2)
        if length<100:
            continue


        if -0.3<slope<0.3:
            continue


        if x_mid<width*0.2 or x_mid>width*0.8 or y_mid<height*0.2:
            continue



        if slope < -0.3 and x_mid < mid_x:
            left_lines.append((slope, intercept))

        elif slope > 0.3 and x_mid > mid_x:
            right_lines.append((slope, intercept))

    left_avg = np.mean(left_lines, axis=0) if left_lines else None
    right_avg = np.mean(right_lines, axis=0) if right_lines else None

    y1 = height
    y2 = int(height * 0.6)

    left_line = make_line_points(y1, y2, left_avg) if left_avg is not None else None
    right_line = make_line_points(y1, y2, right_avg) if right_avg is not None else None

    return left_line, right_line


def smooth_line(line, history):
    if line is not None:
        history.append(line)

    if len(history) == 0:
        return None
    
    # ignore new line if not similar
    avg=np.mean(history,axis=0)

    if line is not None:
        diff=np.linalg.norm(line-avg)
        if diff>100: #prag
            return avg.astype(int)
        
    if line is None:
        return np.mean(history, axis=0).astype(int) if len(history) > 0 else None

    return avg.astype(int)



















# CHANGE VIDEOS HERE:
cap = cv2.VideoCapture("road_video5.mp4")

cv2.namedWindow("Lane Detection Video", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Lane Detection Video", 800, 600)



fourcc = cv2.VideoWriter_fourcc(*'mp4v')

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter("output.mp4", fourcc, fps, (width, height))




while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    height, width, _ = frame.shape

    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    
    edges = cv2.Canny(blur, 100, 250)

    
    mask = np.zeros_like(edges)

    polygon = np.array([[
        (int(width * 0.02), height),
        (int(width * 0.99), height),
        (int(width * 0.7), int(height * 0.6)),
        (int(width * 0.3), int(height * 0.6))
    ]], np.int32)


    cv2.fillPoly(mask, polygon, 255)

    masked_edges = cv2.bitwise_and(edges, mask)


    mask_small=cv2.resize(mask,(500,300))
    masked_edges_small=cv2.resize(masked_edges,(500,300))

    cv2.imshow("mask",mask_small)
    cv2.imshow("masked edges", masked_edges_small)











    
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 50, minLineLength=40, maxLineGap=100)

    
    left_line, right_line = average_slope_intercept(lines, height, width)

    
    left_smooth = smooth_line(left_line, left_history)
    right_smooth = smooth_line(right_line, right_history)


    
    lane_image = np.zeros_like(frame)

    points=None

    if left_smooth is not None and right_smooth is not None:

        points = np.array([[
            (left_smooth[0], left_smooth[1]),   #down left
            (left_smooth[2], left_smooth[3]),   
            (right_smooth[2], right_smooth[3]), # up right
            (right_smooth[0], right_smooth[1])  
        ]], dtype=np.int32)
    
    
    if points is not None:
        cv2.fillPoly(lane_image, points, (0, 255, 0))  




    if left_smooth is not None:
        cv2.line(lane_image,
                 (left_smooth[0], left_smooth[1]),
                 (left_smooth[2], left_smooth[3]),
                 (0, 0, 255), 20)

    if right_smooth is not None:
        cv2.line(lane_image,
                 (right_smooth[0], right_smooth[1]),
                 (right_smooth[2], right_smooth[3]),
                 (255, 0, 0), 20)
        




    
    combo = cv2.addWeighted(frame, 0.8, lane_image, 0.5, 0)

    out.write(combo)

    cv2.imshow("Lane Detection Video", combo)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()