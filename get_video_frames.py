"""Use OpenCV to get each frame of the video.
Or, specify a step parameter to get every nth frame
"""
import cv2

def get_video_frames(video_filepath: str, step=1):
    """Use OpenCV to get each frame of the video.
    Or, specify a step parameter to get every nth frame
    """
    # Load video into videocapture object
    video = cv2.VideoCapture(video_filepath)
    if (video.isOpened()):
        print("Opened " + video_filepath + " successfully.")
    else:
        print("Error opening video: " + video_filepath)
        return
    
    # Get the total number of frames in the video
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    index = []
    count = 0
    keep_reading = True

    while keep_reading:
        if count >= total_frames:
            keep_reading = False
            count = total_frames - 1
        video.set(cv2.CAP_PROP_POS_FRAMES, count)
        print("Reading frame " + str(count) + " of " + str(int (total_frames - 1)))
        success, image = video.read()
        frames.append(image)
        index.append(count)
        count += step
    
    video.release()
    return frames, index



if __name__ == "__main__":
    frames, index = get_video_frames("two_hand.mp4", 10)
    for i in range(len(frames)):
        cv2.imshow("Frame " + str(index[i]),frames[i])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
