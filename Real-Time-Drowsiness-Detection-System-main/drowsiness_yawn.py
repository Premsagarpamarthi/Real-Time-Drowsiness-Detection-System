from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import playsound
import os

def sound_alarm(path):
    global alarm_status
    global alarm_status2
    global saying
    
    try:
        if not os.path.exists(path):
            print(f"Error: Alarm file not found at {os.path.abspath(path)}")
            return
            
        while alarm_status:
            print(f'Playing alarm from: {os.path.abspath(path)}')
            playsound.playsound(path)
            
        if alarm_status2:
            print(f'Playing alarm from: {os.path.abspath(path)}')
            saying = True
            playsound.playsound(path)
            saying = False
    except Exception as e:
        print(f"Error playing alarm sound: {str(e)}")

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def final_ear(shape):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]

    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)

    ear = (leftEAR + rightEAR) / 2.0
    return (ear, leftEye, rightEye)

# Parse command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--webcam", type=int, default=0, help="index of webcam on system")
ap.add_argument("-a", "--alarm", type=str, default="Alert.wav", help="path alarm .wav file")
args = vars(ap.parse_args())

# Initialize the VideoStream
print("[INFO] Starting video stream thread...")
vs = VideoStream(src=args["webcam"]).start()
time.sleep(2.0)

# Initialize the face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize the alarm status
alarm_status = False
alarm_status2 = False
saying = False

# Main loop for processing frames
while True:
    frame = vs.read()
    if frame is None:
        print("Error: Could not read frame from webcam")
        break
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    rects = detector(gray, 0)
    
    # Process each detected face
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        # Calculate the eye aspect ratio
        ear, leftEye, rightEye = final_ear(shape)
        
        # If the eye aspect ratio is below a certain threshold, trigger the alarm
        if ear < 0.3:
            alarm_status = True
            sound_alarm(args["alarm"])
        else:
            alarm_status = False
    
    # Display the frame
    cv2.imshow('Webcam and the drowsiness detection system', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
print("[INFO] Cleaning up...")
cv2.destroyAllWindows()
vs.stop()

# Force close all OpenCV windows
for i in range(1, 5):
    cv2.waitKey(1)

# Initialize the VideoStream
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Initialize the face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize the alarm status
alarm_status = False
alarm_status2 = False
saying = False

# Main loop for processing frames
while True:
    frame = vs.read()
    if frame is None:
        print("Error: Could not read frame from webcam")
        break
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    rects = detector(gray, 0)
    
    # Process each detected face
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        # Calculate the eye aspect ratio
        ear, leftEye, rightEye = final_ear(shape)  # Unpack the tuple
        
        # If the eye aspect ratio is below a certain threshold, trigger the alarm
        if ear < 0.3:
            alarm_status = True
            sound_alarm("Alert.wav")
        else:
            alarm_status = False
    
    # Display the frame
    cv2.imshow('Webcam and the drowsiness detection system', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
vs.stop()

# Initialize the VideoStream
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Initialize the face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize the alarm status
alarm_status = False
alarm_status2 = False
saying = False

# Main loop for processing frames
while True:
    frame = vs.read()
    if frame is None:
        print("Error: Could not read frame from webcam")
        break
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    rects = detector(gray, 0)
    
    # Process each detected face
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        # Calculate the eye aspect ratio
        ear, leftEye, rightEye = final_ear(shape)  # Unpack the tuple
        
        # If the eye aspect ratio is below a certain threshold, trigger the alarm
        if ear < 0.3:
            alarm_status = True
            sound_alarm("Alert.wav")
        else:
            alarm_status = False
    
    # Display the frame
    cv2.imshow('Webcam and the drowsiness detection system', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
vs.stop()

# Initialize the VideoStream
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Initialize the face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize the alarm status
alarm_status = False
alarm_status2 = False
saying = False

# Main loop for processing frames
while True:
    frame = vs.read()
    if frame is None:
        print("Error: Could not read frame from webcam")
        break
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    rects = detector(gray, 0)
    
    # Process each detected face
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        # Calculate the eye aspect ratio
        ear, leftEye, rightEye = final_ear(shape)  # Unpack the tuple
        
        # If the eye aspect ratio is below a certain threshold, trigger the alarm
        if ear < 0.3:
            alarm_status = True
            sound_alarm("Alert.wav")
        else:
            alarm_status = False
    
    # Display the frame
    cv2.imshow('Webcam and the drowsiness detection system', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
vs.stop()

# Initialize the VideoStream
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Initialize the face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize the alarm status
alarm_status = False
alarm_status2 = False
saying = False

# Main loop for processing frames
while True:
    frame = vs.read()
    if frame is None:
        print("Error: Could not read frame from webcam")
        break
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    rects = detector(gray, 0)
    
    # Process each detected face
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        # Calculate the eye aspect ratio
        ear, leftEye, rightEye = final_ear(shape)  # Unpack the tuple
        
        # If the eye aspect ratio is below a certain threshold, trigger the alarm
        if ear < 0.3:
            alarm_status = True
            sound_alarm("Alert.wav")
        else:
            alarm_status = False
    
    # Display the frame
    cv2.imshow('Webcam and the drowsiness detection system', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
vs.stop()

# Initialize the VideoStream
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Initialize the face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize the alarm status
alarm_status = False
alarm_status2 = False
saying = False

# Main loop for processing frames
while True:
    frame = vs.read()
    if frame is None:
        print("Error: Could not read frame from webcam")
        break
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    rects = detector(gray, 0)
    
    # Process each detected face
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        # Calculate the eye aspect ratio
        ear, leftEye, rightEye = final_ear(shape)  # Unpack the tuple
        
        # If the eye aspect ratio is below a certain threshold, trigger the alarm
        if ear < 0.3:
            alarm_status = True
            sound_alarm("Alert.wav")
        else:
            alarm_status = False
    
    # Display the frame
    cv2.imshow('Webcam and the drowsiness detection system', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
vs.stop()

# Initialize the VideoStream
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Initialize the face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize the alarm status
alarm_status = False
alarm_status2 = False
saying = False

# Main loop for processing frames
while True:
    frame = vs.read()
    if frame is None:
        print("Error: Could not read frame from webcam")
        break
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    rects = detector(gray, 0)
    
    # Process each detected face
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        # Calculate the eye aspect ratio
        ear, leftEye, rightEye = final_ear(shape)  # Unpack the tuple
        
        # If the eye aspect ratio is below a certain threshold, trigger the alarm
        if ear < 0.3:
            alarm_status = True
            sound_alarm("Alert.wav")
        else:
            alarm_status = False
    
    # Display the frame
    cv2.imshow('Webcam and the drowsiness detection system', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
vs.stop()

# Initialize the VideoStream
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Initialize the face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize the alarm status
alarm_status = False
alarm_status2 = False
saying = False

# Main loop for processing frames
while True:
    frame = vs.read()
    if frame is None:
        print("Error: Could not read frame from webcam")
        break
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    rects = detector(gray, 0)
    
    # Process each detected face
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        # Calculate the eye aspect ratio
        ear, leftEye, rightEye = final_ear(shape)  # Unpack the tuple
        
        # If the eye aspect ratio is below a certain threshold, trigger the alarm
        if ear < 0.3:
            alarm_status = True
            sound_alarm("Alert.wav")
        else:
            alarm_status = False
    
    # Display the frame
    cv2.imshow('Webcam and the drowsiness detection system', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
vs.stop()

# Initialize the VideoStream
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Initialize the face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize the alarm status
alarm_status = False
alarm_status2 = False
saying = False

# Main loop for processing frames
while True:
    frame = vs.read()
    if frame is None:
        print("Error: Could not read frame from webcam")
        break
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    rects = detector(gray, 0)
    
    # Process each detected face
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        # Calculate the eye aspect ratio
        ear, leftEye, rightEye = final_ear(shape)  # Unpack the tuple
        
        # If the eye aspect ratio is below a certain threshold, trigger the alarm
        if ear < 0.3:
            alarm_status = True
            sound_alarm("Alert.wav")
        else:
            alarm_status = False
    
    # Display the frame
    cv2.imshow('Webcam and the drowsiness detection system', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
vs.stop()

# Initialize the VideoStream
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Initialize the face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize the alarm status
alarm_status = False
alarm_status2 = False
saying = False

# Main loop for processing frames
while True:
    frame = vs.read()
    if frame is None:
        print("Error: Could not read frame from webcam")
        break
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    rects = detector(gray, 0)
    
    # Process each detected face
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        # Calculate the eye aspect ratio
        ear, leftEye, rightEye = final_ear(shape)  # Unpack the tuple
        
        # If the eye aspect ratio is below a certain threshold, trigger the alarm
        if ear < 0.3:
            alarm_status = True
            sound_alarm("Alert.wav")
        else:
            alarm_status = False
    
    # Display the frame
    cv2.imshow('Webcam and the drowsiness detection system', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
vs.stop()

# Initialize the VideoStream
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Initialize the face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize the alarm status
alarm_status = False
alarm_status2 = False
saying = False

# Main loop for processing frames
while True:
    frame = vs.read()
    if frame is None:
        print("Error: Could not read frame from webcam")
        break
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    rects = detector(gray, 0)
    
    # Process each detected face
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        # Calculate the eye aspect ratio
        ear, leftEye, rightEye = final_ear(shape)  # Unpack the tuple
        
        # If the eye aspect ratio is below a certain threshold, trigger the alarm
        if ear < 0.3:
            alarm_status = True
            sound_alarm("Alert.wav")
        else:
            alarm_status = False
    
    # Display the frame
    cv2.imshow('Webcam and the drowsiness detection system', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
vs.stop()

# Initialize the VideoStream
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Initialize the face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize the alarm status
alarm_status = False
alarm_status2 = False
saying = False

# Main loop for processing frames
while True:
    frame = vs.read()
    if frame is None:
        print("Error: Could not read frame from webcam")
        break
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    rects = detector(gray, 0)
    
    # Process each detected face
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        # Calculate the eye aspect ratio
        ear, leftEye, rightEye = final_ear(shape)  # Unpack the tuple
        
        # If the eye aspect ratio is below a certain threshold, trigger the alarm
        if ear < 0.3:
            alarm_status = True
            sound_alarm("Alert.wav")
        else:
            alarm_status = False
    
    # Display the frame
    cv2.imshow('Webcam and the drowsiness detection system', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
vs.stop()

# Initialize the VideoStream
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Initialize the face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize the alarm status
alarm_status = False
alarm_status2 = False
saying = False

# Main loop for processing frames
while True:
    frame = vs.read()
    if frame is None:
        print("Error: Could not read frame from webcam")
        break
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    rects = detector(gray, 0)
    
    # Process each detected face
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        # Calculate the eye aspect ratio
        ear, leftEye, rightEye = final_ear(shape)  # Unpack the tuple
        
        # If the eye aspect ratio is below a certain threshold, trigger the alarm
        if ear < 0.3:
            alarm_status = True
            sound_alarm("Alert.wav")
        else:
            alarm_status = False
    
    # Display the frame
    cv2.imshow('Webcam and the drowsiness detection system', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
vs.stop()

# Initialize the VideoStream
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Initialize the face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize the alarm status
alarm_status = False
alarm_status2 = False
saying = False

# Main loop for processing frames
while True:
    frame = vs.read()
    if frame is None:
        print("Error: Could not read frame from webcam")
        break
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    rects = detector(gray, 0)
    
    # Process each detected face
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        # Calculate the eye aspect ratio
        ear, leftEye, rightEye = final_ear(shape)  # Unpack the tuple
        
        # If the eye aspect ratio is below a certain threshold, trigger the alarm
        if ear < 0.3:
            alarm_status = True
            sound_alarm("Alert.wav")
        else:
            alarm_status = False
    
    # Display the frame
    cv2.imshow('Webcam and the drowsiness detection system', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
vs.stop()

# Initialize the VideoStream
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Initialize the face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize the alarm status
alarm_status = False
alarm_status2 = False
saying = False

# Main loop for processing frames
while True:
    frame = vs.read()
    if frame is None:
        print("Error: Could not read frame from webcam")
        break
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    rects = detector(gray, 0)
    
    # Process each detected face
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        # Calculate the eye aspect ratio
        ear, leftEye, rightEye = final_ear(shape)  # Unpack the tuple
        
        # If the eye aspect ratio is below a certain threshold, trigger the alarm
        if ear < 0.3:
            alarm_status = True
            sound_alarm("Alert.wav")
        else:
            alarm_status = False
    
    # Display the frame
    cv2.imshow('Webcam and the drowsiness detection system', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
vs.stop()

# Initialize the VideoStream
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Initialize the face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize the alarm status
alarm_status = False
alarm_status2 = False
saying = False

# Main loop for processing frames
while True:
    frame = vs.read()
    if frame is None:
        print("Error: Could not read frame from webcam")
        break
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    rects = detector(gray, 0)
    
    # Process each detected face
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        # Calculate the eye aspect ratio
        ear, leftEye, rightEye = final_ear(shape)  # Unpack the tuple
        
        # If the eye aspect ratio is below a certain threshold, trigger the alarm
        if ear < 0.3:
            alarm_status = True
            sound_alarm("Alert.wav")
        else:
            alarm_status = False
    
    # Display the frame
    cv2.imshow('Webcam and the drowsiness detection system', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
vs.stop()

# Initialize the VideoStream
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Initialize the face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize the alarm status
alarm_status = False
alarm_status2 = False
saying = False

# Main loop for processing frames
while True:
    frame = vs.read()
    if frame is None:
        print("Error: Could not read frame from webcam")
        break
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    rects = detector(gray, 0)
    
    # Process each detected face
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        # Calculate the eye aspect ratio
        ear, leftEye, rightEye = final_ear(shape)  # Unpack the tuple
        
        # If the eye aspect ratio is below a certain threshold, trigger the alarm
        if ear < 0.3:
            alarm_status = True
            sound_alarm("Alert.wav")
        else:
            alarm_status = False
    
    # Display the frame
    cv2.imshow('Webcam and the drowsiness detection system', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
vs.stop()

# Initialize the VideoStream
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Initialize the face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize the alarm status
alarm_status = False
alarm_status2 = False
saying = False

# Main loop for processing frames
while True:
    frame = vs.read()
    if frame is None:
        print("Error: Could not read frame from webcam")
        break
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    rects = detector(gray, 0)
    
    # Process each detected face
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        # Calculate the eye aspect ratio
        ear, leftEye, rightEye = final_ear(shape)  # Unpack the tuple
        
        # If the eye aspect ratio is below a certain threshold, trigger the alarm
        if ear < 0.3:
            alarm_status = True
            sound_alarm("Alert.wav")
        else:
            alarm_status = False
    
    # Display the frame
    cv2.imshow('Webcam and the drowsiness detection system', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
vs.stop()

# Initialize the VideoStream
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Initialize the face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize the alarm status
alarm_status = False
alarm_status2 = False
saying = False

# Main loop for processing frames
while True:
    frame = vs.read()
    if frame is None:
        print("Error: Could not read frame from webcam")
        break
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    rects = detector(gray, 0)
    
    # Process each detected face
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        # Calculate the eye aspect ratio
        ear, leftEye, rightEye = final_ear(shape)  # Unpack the tuple
        
        # If the eye aspect ratio is below a certain threshold, trigger the alarm
        if ear < 0.3:
            alarm_status = True
            sound_alarm("Alert.wav")
        else:
            alarm_status = False
    
    # Display the frame
    cv2.imshow('Webcam and the drowsiness detection system', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
vs.stop()

# Initialize the VideoStream
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Initialize the face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize the alarm status
alarm_status = False
alarm_status2 = False
saying = False

# Main loop for processing frames
while True:
    frame = vs.read()
    if frame is None:
        print("Error: Could not read frame from webcam")
        break
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    rects = detector(gray, 0)
    
    # Process each detected face
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        # Calculate the eye aspect ratio
        ear, leftEye, rightEye = final_ear(shape)  # Unpack the tuple
        
        # If the eye aspect ratio is below a certain threshold, trigger the alarm
        if ear < 0.3:
            alarm_status = True
            sound_alarm("Alert.wav")
        else:
            alarm_status = False
    
    # Display the frame
    cv2.imshow('Webcam and the drowsiness detection system', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
vs.stop()

# Initialize the VideoStream
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Initialize the face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize the alarm status
alarm_status = False
alarm_status2 = False
saying = False

# Main loop for processing frames
while True:
    frame = vs.read()
    if frame is None:
        print("Error: Could not read frame from webcam")
        break
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    rects = detector(gray, 0)
    
    # Process each detected face
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        # Calculate the eye aspect ratio
        ear, leftEye, rightEye = final_ear(shape)  # Unpack the tuple
        
        # If the eye aspect ratio is below a certain threshold, trigger the alarm
        if ear < 0.3:
            alarm_status = True
            sound_alarm("Alert.wav")
        else:
            alarm_status = False
    
    # Display the frame
    cv2.imshow('Webcam and the drowsiness detection system', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
vs.stop()

# Initialize the VideoStream
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Initialize the face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize the alarm status
alarm_status = False
alarm_status2 = False
saying = False

# Main loop for processing frames
while True:
    frame = vs.read()
    if frame is None:
        print("Error: Could not read frame from webcam")
        break
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    rects = detector(gray, 0)
    
    # Process each detected face
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        # Calculate the eye aspect ratio
        ear, leftEye, rightEye = final_ear(shape)  # Unpack the tuple
        
        # If the eye aspect ratio is below a certain threshold, trigger the alarm
        if ear < 0.3:
            alarm_status = True
            sound_alarm("Alert.wav")
        else:
            alarm_status = False
    
    # Display the frame
    cv2.imshow('Webcam and the drowsiness detection system', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
vs.stop()

# Initialize the VideoStream
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Initialize the face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize the alarm status
alarm_status = False
alarm_status2 = False
saying = False

# Main loop for processing frames
while True:
    frame = vs.read()
    if frame is None:
        print("Error: Could not read frame from webcam")
        break
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    rects = detector(gray, 0)
    
    # Process each detected face
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        # Calculate the eye aspect ratio
        ear, leftEye, rightEye = final_ear(shape)  # Unpack the tuple
        
        # If the eye aspect ratio is below a certain threshold, trigger the alarm
        if ear < 0.3:
            alarm_status = True
            sound_alarm("Alert.wav")
        else:
            alarm_status = False
    
    # Display the frame
    cv2.imshow('Webcam and the drowsiness detection system', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
vs.stop()

# Initialize the VideoStream
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Initialize the face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize the alarm status
alarm_status = False
alarm_status2 = False
saying = False

# Main loop for processing frames
while True:
    frame = vs.read()
    if frame is None:
        print("Error: Could not read frame from webcam")
        break
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    rects = detector(gray, 0)
    
    # Process each detected face
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        # Calculate the eye aspect ratio
        ear, leftEye, rightEye = final_ear(shape)  # Unpack the tuple
        
        # If the eye aspect ratio is below a certain threshold, trigger the alarm
        if ear < 0.3:
            alarm_status = True
            sound_alarm("Alert.wav")
        else:
            alarm_status = False
    
    # Display the frame
    cv2.imshow('Webcam and the drowsiness detection system', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
vs.stop()

# Initialize the VideoStream
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Initialize the face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize the alarm status
alarm_status = False
alarm_status2 = False
saying = False

# Main loop for processing frames
while True:
    frame = vs.read()
    if frame is None:
        print("Error: Could not read frame from webcam")
        break
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    rects = detector(gray, 0)
    
    # Process each detected face
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        # Calculate the eye aspect ratio
        ear, leftEye, rightEye = final_ear(shape)  # Unpack the tuple
        
        # If the eye aspect ratio is below a certain threshold, trigger the alarm
        if ear < 0.3:
            alarm_status = True
            sound_alarm("Alert.wav")
        else:
            alarm_status = False
    
    # Display the frame
    cv2.imshow('Webcam and the drowsiness detection system', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
vs.stop()

# Initialize the VideoStream
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Initialize the face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize the alarm status
alarm_status = False
alarm_status2 = False
saying = False

# Main loop for processing frames
while True:
    frame = vs.read()
    if frame is None:
        print("Error: Could not read frame from webcam")
        break
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    rects = detector(gray, 0)
    
    # Process each detected face
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        # Calculate the eye aspect ratio
        ear, leftEye, rightEye = final_ear(shape)  # Unpack the tuple
        
        # If the eye aspect ratio is below a certain threshold, trigger the alarm
        if ear < 0.3:
            alarm_status = True
            sound_alarm("Alert.wav")
        else:
            alarm_status = False
    
    # Display the frame
    cv2.imshow('Webcam and the drowsiness detection system', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
vs.stop()

# Initialize the VideoStream
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Initialize the face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize the alarm status
alarm_status = False
alarm_status2 = False
saying = False

# Main loop for processing frames
while True:
    frame = vs.read()
    if frame is None:
        print("Error: Could not read frame from webcam")
        break
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    rects = detector(gray, 0)
    
    # Process each detected face
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        # Calculate the eye aspect ratio
        ear, leftEye, rightEye = final_ear(shape)  # Unpack the tuple
        
        # If the eye aspect ratio is below a certain threshold, trigger the alarm
        if ear < 0.3:
            alarm_status = True
            sound_alarm("Alert.wav")
        else:
            alarm_status = False
    
    # Display the frame
    cv2.imshow('Webcam and the drowsiness detection system', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
vs.stop()

# Initialize the VideoStream
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Initialize the face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize the alarm status
alarm_status = False
alarm_status2 = False
saying = False

# Main loop for processing frames
while True:
    frame = vs.read()
    if frame is None:
        print("Error: Could not read frame from webcam")
        break
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    rects = detector(gray, 0)
    
    # Process each detected face
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        # Calculate the eye aspect ratio
        ear, leftEye, rightEye = final_ear(shape)  # Unpack the tuple
        
        # If the eye aspect ratio is below a certain threshold, trigger the alarm
        if ear < 0.3:
            alarm_status = True
            sound_alarm("Alert.wav")
        else:
            alarm_status = False
    
    # Display the frame
    cv2.imshow('Webcam and the drowsiness detection system', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
vs.stop()

# Initialize the VideoStream
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Initialize the face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize the alarm status
alarm_status = False
alarm_status2 = False
saying = False

# Main loop for processing frames
while True:
    frame = vs.read()
    if frame is None:
        print("Error: Could not read frame from webcam")
        break
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    rects = detector(gray, 0)
    
    # Process each detected face
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        # Calculate the eye aspect ratio
        ear, leftEye, rightEye = final_ear(shape)  # Unpack the tuple
        
        # If the eye aspect ratio is below a certain threshold, trigger the alarm
        if ear < 0.3:
            alarm_status = True
            sound_alarm("Alert.wav")
        else:
            alarm_status = False
    
    # Display the frame
    cv2.imshow('Webcam and the drowsiness detection system', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
vs.stop()

# Initialize the VideoStream
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Initialize the face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize the alarm status
alarm_status = False
alarm_status2 = False
saying = False

# Main loop for processing frames
while True:
    frame = vs.read()
    if frame is None:
        print("Error: Could not read frame from webcam")
        break
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    rects = detector(gray, 0)
    
    # Process each detected face
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        # Calculate the eye aspect ratio
        ear, leftEye, rightEye = final_ear(shape)  # Unpack the tuple
        
        # If the eye aspect ratio is below a certain threshold, trigger the alarm
        if ear < 0.3:
            alarm_status = True
            sound_alarm("Alert.wav")
        else:
            alarm_status = False
    
    # Display the frame
    cv2.imshow('Webcam and the drowsiness detection system', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
vs.stop()

# Initialize the VideoStream
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Initialize the face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize the alarm status
alarm_status = False
alarm_status2 = False
saying = False

# Main loop for processing frames
while True:
    frame = vs.read()
    if frame is None:
        print("Error: Could not read frame from webcam")
        break
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    rects = detector(gray, 0)
    
    # Process each detected face
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        # Calculate the eye aspect ratio
        ear, leftEye, rightEye = final_ear(shape)  # Unpack the tuple
        
        # If the eye aspect ratio is below a certain threshold, trigger the alarm
        if ear < 0.3:
            alarm_status = True
            sound_alarm("Alert.wav")
        else:
            alarm_status = False
    
    # Display the frame
    cv2.imshow('Webcam and the drowsiness detection system', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
vs.stop()

# Initialize the VideoStream
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Initialize the face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize the alarm status
alarm_status = False
alarm_status2 = False
saying = False

# Main loop for processing frames
while True:
    frame = vs.read()
    if frame is None:
        print("Error: Could not read frame from webcam")
        break
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    rects = detector(gray, 0)
    
    # Process each detected face
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        # Calculate the eye aspect ratio
        ear, leftEye, rightEye = final_ear(shape)  # Unpack the tuple
        
        # If the eye aspect ratio is below a certain threshold, trigger the alarm
        if ear < 0.3:
            alarm_status = True
            sound_alarm("Alert.wav")
        else:
            alarm_status = False
    
    # Display the frame
    cv2.imshow('Webcam and the drowsiness detection system', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
vs.stop()

# Initialize the VideoStream
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Initialize the face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize the alarm status
alarm_status = False
alarm_status2 = False
saying = False

# Main loop for processing frames
while True:
    frame = vs.read()
    if frame is None:
        print("Error: Could not read frame from webcam")
        break
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    rects = detector(gray, 0)
    
    # Process each detected face
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        # Calculate the eye aspect ratio
        ear, leftEye, rightEye = final_ear(shape)  # Unpack the tuple
        
        # If the eye aspect ratio is below a certain threshold, trigger the alarm
        if ear < 0.3:
            alarm_status = True
            sound_alarm("Alert.wav")
        else:
            alarm_status = False
    
    # Display the frame
    cv2.imshow('Webcam and the drowsiness detection system', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
vs.stop()

# Initialize the VideoStream
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Initialize the face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize the alarm status
alarm_status = False
alarm_status2 = False
saying = False

# Main loop for processing frames
while True:
    frame = vs.read()
    if frame is None:
        print("Error: Could not read frame from webcam")
        break
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    rects = detector(gray, 0)
    
    # Process each detected face
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        # Calculate the eye aspect ratio
        ear, leftEye, rightEye = final_ear(shape)  # Unpack the tuple
        
        # If the eye aspect ratio is below a certain threshold, trigger the alarm
        if ear < 0.3:
            alarm_status = True
            sound_alarm("Alert.wav")
        else:
            alarm_status = False
    
    # Display the frame
    cv2.imshow('Webcam and the drowsiness detection system', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
vs.stop()

# Initialize the VideoStream
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Initialize the face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize the alarm status
alarm_status = False
alarm_status2 = False
saying = False

# Main loop for processing frames
while True:
    frame = vs.read()
    if frame is None:
        print("Error: Could not read frame from webcam")
        break
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    rects = detector(gray, 0)
    
    # Process each detected face
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        # Calculate the eye aspect ratio
        ear, leftEye, rightEye = final_ear(shape)  # Unpack the tuple
        
        # If the eye aspect ratio is below a certain threshold, trigger the alarm
        if ear < 0.3:
            alarm_status = True
            sound_alarm("Alert.wav")
        else:
            alarm_status = False
    
    # Display the frame
    cv2.imshow('Webcam and the drowsiness detection system', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
vs.stop()

# Initialize the VideoStream
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Initialize the face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize the alarm status
alarm_status = False
alarm_status2 = False
saying = False

# Main loop for processing frames
while True:
    frame = vs.read()
    if frame is None:
        print("Error: Could not read frame from webcam")
        break
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    rects = detector(gray, 0)
    
    # Process each detected face
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        # Calculate the eye aspect ratio
        ear, leftEye, rightEye = final_ear(shape)  # Unpack the tuple
        
        # If the eye aspect ratio is below a certain threshold, trigger the alarm
        if ear < 0.3:
            alarm_status = True
            sound_alarm("Alert.wav")
        else:
            alarm_status = False
    
    # Display the frame
    cv2.imshow('Webcam and the drowsiness detection system', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
vs.stop()

# Initialize the VideoStream
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Initialize the face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize the alarm status
alarm_status = False
alarm_status2 = False
saying = False

# Main loop for processing frames
while True:
    frame = vs.read()
    if frame is None:
        print("Error: Could not read frame from webcam")
        break
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    rects = detector(gray, 0)
    
    # Process each detected face
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        # Calculate the eye aspect ratio
        ear, leftEye, rightEye = final_ear(shape)  # Unpack the tuple
        
        # If the eye aspect ratio is below a certain threshold, trigger the alarm
        if ear < 0.3:
            alarm_status = True
            sound_alarm("Alert.wav")
        else:
            alarm_status = False
    
    # Display the frame
    cv2.imshow('Webcam and the drowsiness detection system', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
vs.stop()

# Initialize the VideoStream
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Initialize the face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize the alarm status
alarm_status = False
alarm_status2 = False
saying = False

# Main loop for processing frames
while True:
    frame = vs.read()
    if frame is None:
        print("Error: Could not read frame from webcam")
        break
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    rects = detector(gray, 0)
    
    # Process each detected face
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        # Calculate the eye aspect ratio
        ear, leftEye, rightEye = final_ear(shape)  # Unpack the tuple
        
        # If the eye aspect ratio is below a certain threshold, trigger the alarm
        if ear < 0.3:
            alarm_status = True
            sound_alarm("Alert.wav")
        else:
            alarm_status = False
    
    # Display the frame
    cv2.imshow('Webcam and the drowsiness detection system', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
vs.stop()

# Initialize the VideoStream
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Initialize the face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize the alarm status
alarm_status = False
alarm_status2 = False
saying = False

# Main loop for processing frames
while True:
    frame = vs.read()
    if frame is None:
        print("Error: Could not read frame from webcam")
        break
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    rects = detector(gray, 0)
    
    # Process each detected face
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        # Calculate the eye aspect ratio
        ear, leftEye, rightEye = final_ear(shape)  # Unpack the tuple
        
        # If the eye aspect ratio is below a certain threshold, trigger the alarm
        if ear < 0.3:
            alarm_status = True
            sound_alarm("Alert.wav")
        else:
            alarm_status = False
    
    # Display the frame
    cv2.imshow('Webcam and the drowsiness detection system', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
vs.stop()

# Initialize the VideoStream
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Initialize the face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize the alarm status
alarm_status = False
alarm_status2 = False
saying = False

# Main loop for processing frames
while True:
    frame = vs.read()
    if frame is None:
        print("Error: Could not read frame from webcam")
        break
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    rects = detector(gray, 0)
    
    # Process each detected face
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        # Calculate the eye aspect ratio
        ear, leftEye, rightEye = final_ear(shape)  # Unpack the tuple
        
        # If the eye aspect ratio is below a certain threshold, trigger the alarm
        if ear < 0.3:
            alarm_status = True
            sound_alarm("Alert.wav")
        else:
            alarm_status = False
    
    # Display the frame
    cv2.imshow('Webcam and the drowsiness detection system', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
vs.stop()

# Initialize the VideoStream
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Initialize the face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize the alarm status
alarm_status = False
alarm_status2 = False
saying = False

# Main loop for processing frames
while True:
    frame = vs.read()
    if frame is None:
        print("Error: Could not read frame from webcam")
        break
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    rects = detector(gray, 0)
    
    # Process each detected face
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        # Calculate the eye aspect ratio
        ear, leftEye, rightEye = final_ear(shape)  # Unpack the tuple
        
        # If the eye aspect ratio is below a certain threshold, trigger the alarm
        if ear < 0.3:
            alarm_status = True
            sound_alarm("Alert.wav")
        else:
            alarm_status = False
    
    # Display the frame
    cv2.imshow('Webcam and the drowsiness detection system', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
vs.stop()

# Initialize the VideoStream
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Initialize the face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize the alarm status
alarm_status = False
alarm_status2 = False
saying = False

# Main loop for processing frames
while True:
    frame = vs.read()
    if frame is None:
        print("Error: Could not read frame from webcam")
        break
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    rects = detector(gray, 0)
    
    # Process each detected face
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        # Calculate the eye aspect ratio
        ear, leftEye, rightEye = final_ear(shape)  # Unpack the tuple
        
        # If the eye aspect ratio is below a certain threshold, trigger the alarm
        if ear < 0.3:
            alarm_status = True
            sound_alarm("Alert.wav")
        else:
            alarm_status = False
    
    # Display the frame
    cv2.imshow('Webcam and the drowsiness detection system', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
vs.stop()

# Initialize the VideoStream
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Initialize the face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize the alarm status
alarm_status = False
alarm_status2 = False
saying = False

# Main loop for processing frames
while True:
    frame = vs.read()
    if frame is None:
        print("Error: Could not read frame from webcam")
        break
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    rects = detector(gray, 0)
    
    # Process each detected face
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        # Calculate the eye aspect ratio
        ear, leftEye, rightEye = final_ear(shape)  # Unpack the tuple
        
        # If the eye aspect ratio is below a certain threshold, trigger the alarm
        if ear < 0.3:
            alarm_status = True
            sound_alarm("Alert.wav")
        else:
            alarm_status = False
    
    # Display the frame
    cv2.imshow('Webcam and the drowsiness detection system', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
vs.stop()

# Initialize the VideoStream
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Initialize the face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize the alarm status
alarm_status = False
alarm_status2 = False
saying = False

# Main loop for processing frames
while True:
    frame = vs.read()
    if frame is None:
        print("Error: Could not read frame from webcam")
        break
    frame = imutils.resize(frame, width=4)
                    