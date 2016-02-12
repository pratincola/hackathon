
import os
import time
import cv2

from picamera.array import PiRGBArray
from picamera import PiCamera
from algorithm import eye_detection, touch_detection




def run_raspi_modules(raspi, pwd):

    face_casc_path = pwd + '/resources/haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(face_casc_path)

    eye_casc_path = pwd + '/resources/eyes.xml'
    eye_cascade = cv2.CascadeClassifier(eye_casc_path)

    if face_cascade.empty():
        raise IOError('Unable to load the face cascade classifier xml file')

    if eye_cascade.empty():
        raise IOError('Unable to load the eye cascade classifier xml file')

    if raspi:
        camera = PiCamera()
        camera.resolution = (640, 480)
        camera.framerate = 32
        camera.iso = 800 # try it out
        rawCapture = PiRGBArray(camera, size=(640, 480))

        # allow the camera to warmup
        time.sleep(0.1)

        # capture frames from the camera
        for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
            # Get touch data
            touch_detection.detect_touch()

            # grab the raw NumPy array representing the image, then initialize the timestamp
            # and occupied/unoccupied text
            image = frame.array

            face_gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_gray_frame = cv2.equalizeHist(face_gray_frame)

            # Algorithm
            eye_detection.eye_detection(face_cascade, eye_cascade, face_gray_frame, frame)

            # Display the resulting frame
            cv2.imshow("Frame", image)

            # clear the stream in preparation for the next frame
            rawCapture.truncate(0)

            # if the `q` key was pressed, break from the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    else:

        video_capture = cv2.VideoCapture(0)

        # Main algo
        while True:
            # Get touch data
            # touch_detection.detect_touch()

            # Capture frame-by-frame
            ret, video_frame = video_capture.read()

            # converting frame/image to gray
            face_gray_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)
            face_gray_frame = cv2.equalizeHist(face_gray_frame)

            # Algorithm
            eye_detection.eye_detection(face_cascade, eye_cascade, face_gray_frame, video_frame)

            # Display the resulting frame
            cv2.imshow('Video', video_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

                # When everything is done, release the capture
        video_capture.release()
        cv2.destroyAllWindows()




