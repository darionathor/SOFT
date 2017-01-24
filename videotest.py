import cv2

path= "videos/video-0.avi"

video=cv2.VideoCapture(path)
print(video.isOpened())