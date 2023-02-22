import cv2
import os

EPISODES=25001
STATS_EVERY=10
def make_video():
    # Set up video codec and output file
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # MP4 codec
    out = cv2.VideoWriter('chat.mp4', fourcc, 60.0, (1200, 900))

    # Loop over the saved chart images and add them to the video
    for i in range(0, EPISODES, STATS_EVERY):
        img_path = f"chat_charts/{i}.png"
        print(img_path)
        frame = cv2.imread(img_path)
        out.write(frame)

    # Release the video file
    out.release()




make_video()