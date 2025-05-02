# python drowsiness_yawn.py --webcam webcam_index
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
import pickle

# At the top of the file, add winsound import
import winsound

# Replace the sound_alarm function with this:
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
            winsound.PlaySound(path, winsound.SND_FILENAME)
            
        if alarm_status2:
            print(f'Playing alarm from: {os.path.abspath(path)}')
            saying = True
            winsound.PlaySound(path, winsound.SND_FILENAME)
            saying = False
    except Exception as e:
        print(f"Error playing alarm sound: {str(e)}")

def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])
    # compute the eye aspect ratio
    ear = float((A + B) / (2.0 * C))
    # return the eye aspect ratio
    return ear

def lip_distance(shape):
    # Extract the top and bottom lip coordinates
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))
    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))
    
    # Calculate mean points for top and bottom lip
    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)
    
    # Calculate the distance
    distance = float(abs(top_mean[1] - low_mean[1]))
    return distance

def calculate_ear(shape):
    # Get facial landmarks for eyes
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    
    # Calculate EAR
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)
    ear = float((leftEAR + rightEAR) / 2.0)
    
    return ear, leftEye, rightEye

# Parse command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--webcam", type=int, default=0,
                help="index of webcam on system")
ap.add_argument("-a", "--alarm", type=str, default="Alert.wav",
                help="path alarm .WAV file")
ap.add_argument("-m", "--model", type=str, default="drowsiness_model.pkl",
                help="path to trained drowsiness detection model")
args = vars(ap.parse_args())

# Check required files
print("[INFO] Checking required files...")
required_files = {
    "cascade": "haarcascade_frontalface_default.xml",
    "predictor": "shape_predictor_68_face_landmarks.dat",
    "alarm": args["alarm"],
    "model": args["model"]
}

for key, filepath in required_files.items():
    if not os.path.exists(filepath):
        print(f"[ERROR] {key} file not found: {filepath}")
        exit(1)

# Initialize the video stream and allow the camera sensor to warm up
print("[INFO] Starting video stream thread...")
vs = VideoStream(src=args["webcam"]).start()
time.sleep(1.0)

# Initialize dlib's face detector and facial landmark predictor
print("[INFO] Loading facial landmark predictor...")
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Load the trained model if provided
print("[INFO] Loading drowsiness detection model...")
model = None
use_trained_model = False
try:
    with open(args["model"], 'rb') as f:
        model = pickle.load(f)
    use_trained_model = True
    print("[INFO] Model loaded successfully")
except Exception as e:
    print(f"[WARNING] Could not load model: {str(e)}")
    print("[INFO] Falling back to threshold-based detection")

# Initialize variables
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 30
YAWN_THRESH = 20
COUNTER = 0
alarm_status = False
alarm_status2 = False
saying = False

print("[INFO] Starting detection...")

while True:
    try:
        # Grab the frame from the threaded video file stream
        frame = vs.read()
        if frame is None:
            print("[ERROR] Could not read frame from webcam")
            break
            
        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the grayscale frame
        faces = detector.detectMultiScale(gray, scaleFactor=1.1,
                                        minNeighbors=5, minSize=(30, 30),
                                        flags=cv2.CASCADE_SCALE_IMAGE)
        
        # Process each detected face
        for (x, y, w, h) in faces:
            rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
            
            # Get facial landmarks
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            
            # Extract features
            ear_value, leftEye, rightEye = calculate_ear(shape)
            mouth_dist = lip_distance(shape)
            
            # Draw the contours around eyes and mouth
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            
            lip = shape[48:60]
            cv2.drawContours(frame, [cv2.convexHull(lip)], -1, (0, 255, 0), 1)
            
            # Use trained model if available
            if use_trained_model:
                features = [ear_value, float(mouth_dist)]
                prediction = model.predict([features])[0]
                
                if prediction == "drowsy":
                    if not alarm_status:
                        alarm_status = True
                        if args["alarm"]:
                            t = Thread(target=sound_alarm, args=(args["alarm"],))
                            t.daemon = True
                            t.start()
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                elif prediction == "yawn":
                    cv2.putText(frame, "Yawn Alert", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    if not alarm_status2 and not saying:
                        alarm_status2 = True
                        if args["alarm"]:
                            t = Thread(target=sound_alarm, args=(args["alarm"],))
                            t.daemon = True
                            t.start()
                else:
                    COUNTER = 0
                    alarm_status = False
                    alarm_status2 = False
                
                cv2.putText(frame, f"State: {prediction}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                # Threshold-based detection
                if ear_value < EYE_AR_THRESH:
                    COUNTER += 1
                    if COUNTER >= EYE_AR_CONSEC_FRAMES:
                        if not alarm_status:
                            alarm_status = True
                            if args["alarm"]:
                                t = Thread(target=sound_alarm, args=(args["alarm"],))
                                t.daemon = True
                                t.start()
                        cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    COUNTER = 0
                    alarm_status = False
                
                if mouth_dist > YAWN_THRESH:
                    cv2.putText(frame, "Yawn Alert", (10, 60),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    if not alarm_status2 and not saying:
                        alarm_status2 = True
                        if args["alarm"]:
                            t = Thread(target=sound_alarm, args=(args["alarm"],))
                            t.daemon = True
                            t.start()
                else:
                    alarm_status2 = False
            
            # Display metrics
            cv2.putText(frame, f"EAR: {ear_value:.2f}", (300, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"YAWN: {float(mouth_dist):.2f}", (300, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Show the frame
        cv2.imshow("Drowsiness Detection", frame)
        
        # Handle key events
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
            
    except Exception as e:
        print(f"[ERROR] An error occurred: {str(e)}")
        continue

# Cleanup
print("[INFO] Cleaning up...")
cv2.destroyAllWindows()
vs.stop()

# Force close all OpenCV windows
for i in range(1, 5):
    cv2.waitKey(1)