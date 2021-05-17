import cv2
import os

path = "D:/NTUT/大二下/多媒體技術與應用/hw07/helmet/"
os.chdir(path)

fname = "helmet3.mp4"
video_capture = cv2.VideoCapture(fname)

ret = False
if video_capture.isOpened():
    ret, frame = video_capture.read()
else:
    print("此影片正在被使用中！")
    print("影片路徑：", os.getcwd(), "\\", fname)

timeFormat = 10
timeCount =  1
pictureCount = 1
while ret:
    ret, frame = video_capture.read()
    if not ret: break

    if(timeCount % timeFormat ==  0):
        picName = ".\\images\\" + "helmet3_" + str(pictureCount) + ".jpg"
        cv2.imwrite(picName, frame)
        print("影片路徑：", picName, sep='')
        
        pictureCount += 1
        timeCount = 0
    timeCount += 1
    #cv2.waitKey(1)

video_capture.release()

