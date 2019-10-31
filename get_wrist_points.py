"""Gets the wrist points from frames[] (output of get_video_frames)
Currently, this function assumes the target person is at index 0.
datum.poseKeypoints outputs a 3 dimensional matrix: (personID, bodypart, (x, y, score))
The left wrist keypoint is index 7 of person's matrix.
OpenPose doesn't identify the wrist if it can't also identify the elbow.
"""

import cv2
import numpy as np
from start_openpose import op, opWrapper
from get_pose_data import get_pose_data
from get_video_frames import get_video_frames
from typing import List

def get_wrist_points(frames: List[np.ndarray], side: str):
    """Gets the wrist points from frames[] (output of get_video_frames)
    Currently, this function assumes the target person is at index 0.
    datum.poseKeypoints outputs a 3 dimensional matrix: (personID, bodypart, (x, y, score))
    The left wrist keypoint is index 7 of person's matrix.
    OpenPose doesn't identify the wrist if it can't also identify the elbow.
    """
    wrist_points = []
    wristID = 4 if side == "right" else 7 # 4 = right, 7 = left
    print(wristID)
    for frame in frames:
        pose_data = get_pose_data(frame)
        wrist_point = pose_data.poseKeypoints[0].reshape((-1, 3))[wristID]
        print(wrist_point)
        wrist_points.append(wrist_point)
    return wrist_points

if __name__ == "__main__":
    frames, index = get_video_frames("two_hand.mp4", 5)
    left_wrist_points = get_wrist_points(frames, "left")
    right_wrist_points = get_wrist_points(frames, "right")
    print(left_wrist_points)
    print(right_wrist_points)
