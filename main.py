__author__ = 'mashers'

import cv2
import os
from algorithm import eye_detection

if __name__ == '__main__':
    pwd = os.path.dirname(os.path.realpath(__file__))
    raspi = False

    face_casc_path = pwd + '/resources/haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(face_casc_path)

    eye_casc_path = pwd + '/resources/eyes.xml'
    eye_cascade = cv2.CascadeClassifier(eye_casc_path)

    if face_cascade.empty():
        raise IOError('Unable to load the face cascade classifier xml file')

    if eye_cascade.empty():
        raise IOError('Unable to load the eye cascade classifier xml file')

    video_capture = cv2.VideoCapture(0)

    # Main algo
    while True:
        # Capture frame-by-frame
        ret, video_frame = video_capture.read()

        # converting frame/image to gray
        face_gray_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)
        face_gray_frame = cv2.equalizeHist(face_gray_frame)

        if raspi:
            pass
        else:
            eye_detection.eye_detection(face_cascade, eye_cascade, face_gray_frame, video_frame)

        # Display the resulting frame
        cv2.imshow('Video', video_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()



