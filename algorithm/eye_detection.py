__author__ = ''

# idea
# 1) look for faces
# 2) if a face is found look for eyes
# 3) if the eye is found store in a variable that eyes are open, else store that eyes are closed
# PROJECT MORPHEUS #########
import cv2


def eye_detection(face_cascade, eye_cascade, face_gray_frame, video_frame):
    # Detects objects of different sizes in the input image. The detected objects are returned as a list of rectangles
    faces = face_cascade.detectMultiScale(
        face_gray_frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(video_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # inside face rectangle detect eyes
        roi_gray = face_gray_frame[y:y+h, x:x+w]
        roi_color = video_frame[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray, 1.03, 5, 0, (40, 40))

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)




