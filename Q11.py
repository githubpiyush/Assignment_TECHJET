import cv2
import os
import bz2
import os
from align import AlignDlib
from urllib.request import urlopen
img_counter = 0
def download_landmarks(dst_file):
    url = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
    decompressor = bz2.BZ2Decompressor()
    
    with urlopen(url) as src, open(dst_file, 'wb') as dst:
        data = src.read(1024)
        while len(data) > 0:
            dst.write(decompressor.decompress(data))
            data = src.read(1024)

dst_dir = 'models'
dst_file = os.path.join(dst_dir, 'landmarks.dat')

if not os.path.exists(dst_file):
    os.makedirs(dst_dir)
    download_landmarks(dst_file)
cam = cv2.VideoCapture(0)

cv2.namedWindow("test")

alignment = AlignDlib('models/landmarks.dat')
# Detect face and return bounding box
path = 'input'
if not os.path.exists(path):
    os.makedirs(path)
while True:
    ret, frame = cam.read()
    if frame is not None:
       cv2.imshow("test", frame)
    if not ret:
        break
    img_name = "opencv_frame_{}.png".format(img_counter)
    faces = alignment.getLargestFaceBoundingBox(frame)
    if faces is not None:
        cv2.imwrite(os.path.join(path , img_name), frame)
        print("{} written!".format(img_name))
        img_counter += 1
    k = cv2.waitKey(1)

    if k%256 == 32:
        # SPACE pressed
        while True:
            ret, frame = cam.read()
            if frame is not None:
                 cv2.imshow("test", frame)
            if not ret:
               break
            l = cv2.waitKey(1)
            if l%256 == 27:
            # esc pressed
                while True:
                      ret, frame = cam.read()
                      if frame is not None:
                           cv2.imshow("test", frame)
                      if not ret:
                          break
                      img_name = "opencv_frame_{}.png".format(img_counter)
                      faces = alignment.getLargestFaceBoundingBox(frame)
                      if faces is not None:
                          cv2.imwrite(os.path.join(path , img_name), frame)
                          print("{} written!".format(img_name))
                          img_counter += 1
                      m = cv2.waitKey(1)
                      if m == ord('s'):
                          cam.release()
                          cv2.destroyAllWindows()
