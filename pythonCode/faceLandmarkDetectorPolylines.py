import cv2
import dlib
import sys
import numpy as np

#draw polylines
def drawPolyline(image, landmarks, start, end, isClosed=False):
        points = []
        for i in range(start, end+1):
                point = [landmarks.part(i).x, landmarks.part(i).y]
                points.append(point)

        points = np.array(points, dtype=np.int32)
        cv2.polylines(image, [points], isClosed, (0, 255, 255), 2, 16)


def drawPolylines(image, landmarks):
        drawPolyline(image, landmarks, 0, 16)           # Jaw line
        drawPolyline(image, landmarks, 17, 21)          # Left eyebrow
        drawPolyline(image, landmarks, 22, 26)          # Right eyebrow
        drawPolyline(image, landmarks, 27, 30)          # Nose bridge
        drawPolyline(image, landmarks, 30, 35, True)    # Lower nose
        drawPolyline(image, landmarks, 36, 41, True)    # Left eye
        drawPolyline(image, landmarks, 42, 47, True)    # Right Eye
        drawPolyline(image, landmarks, 48, 59, True)    # Outer lip
        drawPolyline(image, landmarks, 60, 67, True)    # Inner lip

#Read images
imageSingle = cv2.imread("../assets/anish.jpg")
imageMultiple = cv2.imread("../assets/anish2.jpg")

#create images clone to work on
imageSingleClone = imageSingle.copy()
imageMultipleClone = imageMultiple.copy()

#convert to dlib image format
dlibImageSingle = cv2.cvtColor(imageSingleClone, cv2.COLOR_BGR2RGB)
dlibImageMultiple = cv2.cvtColor(imageMultipleClone, cv2.COLOR_BGR2RGB)

#define face detector
faceDetector = dlib.get_frontal_face_detector()

#define landmark detector and load face landmark model
landmarkDetector = dlib.shape_predictor("../dlibAndModel/shape_predictor_68_face_landmarks.dat")

#detect faces in the images
facesSingle = faceDetector(dlibImageSingle, 0)
facesMultiple = faceDetector(dlibImageMultiple, 0)

#loop over all the faces detected
for i in range(0, len(facesSingle)):
        dlibRectangle = dlib.rectangle(int(facesSingle[i].left()), int(facesSingle[i].top()), int(facesSingle[i].right()), int(facesSingle[i].bottom()))
        
        #for each face run landmark detector
        landmarks = landmarkDetector(dlibImageSingle, dlibRectangle)

        #print number of face landmarks detected
        print("number of face landmarks: ", len(landmarks.parts()))

        #draw polylines on the face
        drawPolylines(imageSingleClone, landmarks)


for i in range(0, len(facesMultiple)):
        dlibRectangle = dlib.rectangle(int(facesMultiple[i].left()), int(facesMultiple[i].top()), int(facesMultiple[i].right()), int(facesMultiple[i].bottom()))
        
        #for each face run landmark detector
        landmarks = landmarkDetector(dlibImageMultiple, dlibRectangle)

        #print number of face landmarks detected
        print("number of face landmarks: ", len(landmarks.parts()))

        #draw polylines on the face
        drawPolylines(imageMultipleClone, landmarks)

#create windows to display images
cv2.namedWindow("single person image", cv2.WINDOW_NORMAL)
cv2.namedWindow("single person landmarks", cv2.WINDOW_NORMAL)
cv2.namedWindow("multiple person image", cv2.WINDOW_NORMAL)
cv2.namedWindow("multiple person landmarks", cv2.WINDOW_NORMAL)

#display images
cv2.imshow("single person image", imageSingle)
cv2.imshow("single person landmarks", imageSingleClone)
cv2.imshow("multiple person image", imageMultiple)
cv2.imshow("multiple person landmarks", imageMultipleClone)

#press esc to exit the program
cv2.waitKey(0)

#close all the opened windows
cv2.destroyAllWindows()