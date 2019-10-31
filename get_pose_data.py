"""Gets pose data from a cv2 image."""
import cv2
import numpy as np
from start_openpose import op, opWrapper

def get_pose_data(cv2image: np.ndarray):
    """Gets pose data from a cv2 image."""
    datum = op.Datum()
    datum.cvInputData = cv2image
    opWrapper.emplaceAndPop([datum])
    return datum

if __name__ == "__main__":
    image = cv2.imread("distracted.jpg")
    datum = get_pose_data(image)

    cv2.imshow("Original Image", image)
    cv2.imshow("Pose Data", datum.cvOutputData)
    cv2.waitKey(0)
    cv2.destroyAllWindows()