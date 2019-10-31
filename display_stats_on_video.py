import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from start_openpose import op, opWrapper
from get_pose_data import get_pose_data
from get_video_frames import get_video_frames
from collections import deque

cap = cv2.VideoCapture("3424.mp4")
# cap = cv2.VideoCapture(0)

def getWristPoint(frame: np.ndarray, side: str):
    wristID = 4 if side == "right" else 7 if side == "left" else None
    pose_data = get_pose_data(frame)
    wrist_point = pose_data.poseKeypoints[0].reshape((-1, 3))[wristID]
    return wrist_point

def showInfo(frame, x, y, text):
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (x, y)
    fontScale              = 0.5
    fontColor              = (255,255,255)
    lineType               = 2

    cv2.putText(frame, text, 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        lineType)

def isInCorner(corner: str, point, box, threshold):
    a = (box["width"] * threshold) / 2
    b = (box["height"] * threshold) / 2
    x = point[0]
    y = point[1]
    h = box[corner][0]
    k = box[corner][1]
    inside = ((x - h) ** 2) / (a ** 2) + ((y - k) ** 2) / (b ** 2) < 1
    return inside

count = 0
count_queue = deque([])
wrist_pos_queue = deque([]) # Keep record of 300 frames
gradient_x_queue = deque([])
gradient_y_queue = deque([])
angle_queue = deque([])
dist_queue = deque([])
queue_length = 300
lookback_length = 10 # look back this number of frames for calculating average speed and speed change
average_speed = 0
min_speed = 0
max_speed = 0
speed_change = 0
corner_region_threshold = 0.96
current_region = ""
found_slow_point = False
slow_point_threshold = 0.1
frame_moment = deque([])
frame_moment_title = deque([])
inflection_corners = []
inflection_text = ""

# Animation functions
# fig, plot = plt.subplots(1, 1, figsize=(10, 10))
# def livePlots(i):
#     # fig, [[plt1, plt2], [plt3, plt4]] = plt.subplots(2, 2, figsize=(20, 20))
#     plot.plot(list(count_queue), list(dist_queue))
#     plot.plot([1, 2, 3, 4, 5], [6, 7, 8, 9, 10])

# ani = animation.FuncAnimation(fig, livePlots, interval=1000)
# plt.show(block=False)

def playVideo():
    # Define globals
    global count
    global count_queue
    global wrist_pos_queue
    global gradient_x_queue
    global gradient_y_queue
    global angle_queue
    global dist_queue
    global queue_length
    global lookback_length
    global average_speed
    global min_speed
    global max_speed
    global speed_change
    global corner_region_threshold
    global current_region
    global found_slow_point
    global frame_moment
    global frame_moment_title
    global inflection_text
    
    # Increase count
    count += 1

    # Capture frame-by-frame
    ret, frame = cap.read()

    if ret is False:
        return

    # Get data for current frame
    wrist_point = getWristPoint(frame, "left")
    wrist_x, wrist_y = wrist_point[0:2]

    if wrist_x == 0 and wrist_y == 0:
        return

    # Draw wrist point onto image
    cv2.circle(frame, (wrist_x, wrist_y), 5, (255,0,0),-1)

    # Add wrist pos to wrist pos queue.
    if len(wrist_pos_queue) == queue_length:
        wrist_pos_queue.popleft()
        gradient_x_queue.popleft()
        gradient_y_queue.popleft()
        angle_queue.popleft()
        dist_queue.popleft()
        count_queue.popleft()
    wrist_pos_queue.append([wrist_x, wrist_y])
    count_queue.append(count)

    # Calculate gradients if queue has enough data
    i = len(wrist_pos_queue) - 1
    if len(wrist_pos_queue) > 1:
        dx = wrist_pos_queue[i][0] - wrist_pos_queue[i - 1][0]
        dy = wrist_pos_queue[i][1] - wrist_pos_queue[i - 1][1]
        gradient_x_queue.append(dx)
        gradient_y_queue.append(dy)
        angle = int(math.atan2(dy, dx) * 180 / math.pi)
        distance = math.sqrt(dx ** 2 + dy ** 2)
        angle_queue.append(angle)
        dist_queue.append(distance)
        # Average speed of last 10 datapoints
        numpy_dist = np.array(list(dist_queue))
        dist_subset = numpy_dist[-lookback_length:].reshape(-1)
        average_speed = np.mean(dist_subset)
        # Min and Max Speed over the entire queue. Not sure how useful this is.
        minmax_subset = numpy_dist[-(int(queue_length / 2)):].reshape(-1)
        min_speed = int(np.amin(minmax_subset).item())
        max_speed = int(np.amax(minmax_subset).item())
        # Change in speed for the last 10 datapoints.
        avg_speed_start = np.mean(dist_subset[:5])
        avg_speed_end = np.mean(dist_subset[-5:])
        speed_change = avg_speed_end - avg_speed_start

    # Calculate bounding box using the recorded points (amount determined by queue_length)
    points = np.array([wrist_pos_queue]).reshape((-1, 2)).astype(int)

    if points[np.sum(points, axis=1) > 0].size > 0:
        xy_max = np.amax(points, axis=0)
        xy_min = np.amin(points[np.sum(points, axis=1) > 0], axis=0) # Ignore (0, 0) points
        cv2.rectangle(frame, (int(xy_min[0]), int(xy_min[1])), (int(xy_max[0]), int(xy_max[1])), (0, 0, 255), 2)
        
        # Bounding box info
        box = {
            "width": xy_max[0] - xy_min[0],
            "height": xy_max[1] - xy_min[1],
            "tl": np.array(xy_min),
            "tr": np.array([xy_max[0], xy_min[1]]),
            "bl": np.array([xy_min[0], xy_max[1]]),
            "br": np.array(xy_max)
        }

        # Detect if current position is near corners.
        inBL = isInCorner("bl", wrist_point, box, corner_region_threshold)
        inBR = isInCorner("br", wrist_point, box, corner_region_threshold)
        inTL = isInCorner("tl", wrist_point, box, corner_region_threshold)
        inTR = isInCorner("tr", wrist_point, box, corner_region_threshold)

        if inBL: current_region = "bl"
        elif inBR: current_region = "br"
        elif inTL: current_region = "tl"
        elif inTR: current_region = "tr"
        else:
            current_region = ""
            found_slow_point = False

        # Draw ellipses for corners
        ellipseWidth = int(box["width"] / 2 * corner_region_threshold)
        ellipseHeight = int(box["height"] / 2 * corner_region_threshold)
        for c in ["tl", "tr", "bl", "br"]:
            cv2.ellipse(frame, (int(box[c][0]), int(box[c][1])), (ellipseWidth, ellipseHeight), 0, 0, 360, (255, 0, 255), 2)

        # Determine if the wrist is slowing down in the region
        if (inBL or inBR or inTL or inTR) and speed_change <= (max_speed * slow_point_threshold) and found_slow_point is False:
            if len(frame_moment) == 3:
                frame_moment.popleft()
                frame_moment_title.popleft()
            frame_moment.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_moment_title.append(str(count))
            ax2.imshow(frame_moment[0])
            ax2.set_title("Frame " + frame_moment_title[0])
            if len(frame_moment) >= 2:
                ax3.imshow(frame_moment[1])
                ax3.set_title("Frame " + frame_moment_title[1])
            if len(frame_moment) >= 3:
                ax4.imshow(frame_moment[2])
                ax4.set_title("Frame " + frame_moment_title[2])
            found_slow_point = True

            # Add inflection corner
            if inBL: inflection_corners.append("bl")
            elif inBR: inflection_corners.append("br")
            elif inTL: inflection_corners.append("tl")
            elif inTR: inflection_corners.append("tr")

    # Check for time signature
    twofour = ["br", "tl"]
    threefour = ["br", "tl", "bl"]
    signature = ""
    for i in range(len(inflection_corners) - 3, 0, -1):
        if inflection_corners[i:i+3] == threefour:
            signature = "threefour"
            break
        if inflection_corners[i:i+2] == twofour:
            signature = "twofour"
            break

    if len(inflection_corners) >= 3:
        inflection_text = inflection_corners[-3] + " " + inflection_corners[-2] + " " + inflection_corners[-1] + "; Signature estimate: " + signature

    # Output Text
    showInfo(frame, 5, 15, "x:" + str(int(wrist_x)) + " y:" + str(int(wrist_y)))

    if len(wrist_pos_queue) > 1:
        last = len(gradient_x_queue) - 1
        showInfo(frame, 5, 35, "Frame Count: " + str(count))
        showInfo(frame, 5, 55, "Movement Angle:" + str(angle_queue[last]))
        showInfo(frame, 5, 75, "Speed:" + str(dist_queue[last]))
        showInfo(frame, 5, 95, "Avg Speed:" + str(average_speed))
        showInfo(frame, 5, 115, "Min Sp:" + str(min_speed) + " MaxSp:" + str(max_speed))
        showInfo(frame, 5, 135, "Speed Change:" + str(speed_change))
        showInfo(frame, 5, 155, "Region:" + current_region)
        showInfo(frame, 5, 175, inflection_text)

    # Display the resulting frame
    cv2.imshow('Wrist Tracking', frame)

    if cv2.waitKey(33) & 0xFF == ord('q'):
        return

# Plot info
fig = plt.figure(figsize=(10, 10))
grid = plt.GridSpec(4, 3, hspace=1, wspace=0.2)
ax1 = fig.add_subplot(grid[0:3, :])
im1, = ax1.plot(np.array(list(count_queue)).reshape(-1), np.array(list(dist_queue)).reshape(-1))
ax1.set_title("Speed")
ax1.set_xlabel("Frame")
ax1.set_ylabel("Speed (px/frame)")
ax2 = fig.add_subplot(grid[3, 0])
ax2.axis("off")
ax3 = fig.add_subplot(grid[3, 1])
ax3.axis("off")
ax4 = fig.add_subplot(grid[3, 2])
ax4.axis("off")

def update(frame):
    playVideo()
    if len(count_queue) > 3:
        count_list = list(count_queue)
        dist_list = list(dist_queue)
        im1.set_data(count_list[:-1], dist_list)
        if len(count_queue) <= lookback_length * 5:
            ax1.set_xlim(0, count_list[-1])
        else:
            ax1.set_xlim(count_list[-(lookback_length * 5)], count_list[-1])
        ax1.set_ylim(np.amin(dist_list), np.amax(dist_list))
        fig.gca().relim()
        fig.gca().autoscale_view()

    return im1,

ani = animation.FuncAnimation(fig, update, interval=1)
plt.show()

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()