# Simple drowsiness detection using laptop webcam (Offline Version)
import cv2
import numpy as np
import dlib
from scipy.spatial import distance as dist
import time
import os
import winsound

# Check required files
if not os.path.exists("shape_predictor_68_face_landmarks.dat"):
    print("[ERROR] Required file 'shape_predictor_68_face_landmarks.dat' not found")
    exit(1)

if not os.path.exists("Alert.wav"):
    print("[ERROR] Required file 'Alert.wav' not found")
    exit(1)

# Initialize face detector and predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Constants for eye detection
EYE_AR_THRESH = 0.18  # Lower threshold for more accurate eye closure detection
EYE_AR_CONSEC_FRAMES = 8  # Fewer frames needed for more responsive detection
EYE_AR_DROWSY_THRESH = 0.22  # Threshold for drowsy state

# Constants for yawn detection
MAR_THRESH = 0.45  # Lower threshold for more accurate yawn detection
YAWN_CONSEC_FRAMES = 6  # Fewer frames needed for more responsive detection
MIN_ALARM_INTERVAL = 2.0  # Minimum time between alarms

# Initialize variables
COUNTER = 0
YAWN_COUNTER = 0
ALARM_ON = False
last_alarm_time = 0
last_state = "AWAKE"

def eye_aspect_ratio(eye):
    # Calculate vertical distances (more precise)
    A = dist.euclidean(eye[1], eye[5])  # Vertical distance 1
    B = dist.euclidean(eye[2], eye[4])  # Vertical distance 2
    # Calculate horizontal distance
    C = dist.euclidean(eye[0], eye[3])  # Horizontal distance
    # Calculate eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth):
    # Calculate vertical distances (more precise)
    A = dist.euclidean(mouth[2], mouth[10])  # Vertical distance 1
    B = dist.euclidean(mouth[4], mouth[8])   # Vertical distance 2
    # Calculate horizontal distance
    C = dist.euclidean(mouth[0], mouth[6])   # Horizontal distance
    # Calculate mouth aspect ratio
    mar = (A + B) / (2.0 * C)
    return mar

def play_alarm():
    global ALARM_ON
    if not ALARM_ON:
        ALARM_ON = True
        try:
            winsound.PlaySound("Alert.wav", winsound.SND_FILENAME | winsound.SND_ASYNC)
            time.sleep(0.5)
        except Exception as e:
            print(f"Error playing alarm: {str(e)}")
        finally:
            ALARM_ON = False

def test_camera(index):
    print(f"[INFO] Testing camera {index}...")
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"[ERROR] Camera {index} not available")
        return False
    
    # Try to read a frame
    ret, frame = cap.read()
    if not ret:
        print(f"[ERROR] Could not read from camera {index}")
        cap.release()
        return False
    
    # Try to display the frame
    try:
        cv2.imshow(f"Camera {index}", frame)
        cv2.waitKey(1000)  # Show for 1 second
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"[ERROR] Could not display frame from camera {index}: {str(e)}")
        cap.release()
        return False
    
    cap.release()
    return True

def main():
    global COUNTER, YAWN_COUNTER, last_alarm_time, last_state
    
    print("[INFO] Starting drowsiness detection...")
    print("[INFO] Press 'q' to quit")
    
    # Try camera 0 (default webcam) first
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open default camera, trying external camera...")
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("[ERROR] No cameras available")
            return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("[INFO] Camera started successfully")
    
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Could not read frame")
            break
        
        # Convert to grayscale and apply histogram equalization
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        # Detect faces with higher confidence
        faces = detector(gray, 1)
        
        current_time = time.time()
        
        for face in faces:
            # Get facial landmarks
            shape = predictor(gray, face)
            shape = np.array([[p.x, p.y] for p in shape.parts()])
            
            # Get eye and mouth coordinates
            leftEye = shape[36:42]
            rightEye = shape[42:48]
            mouth = shape[48:68]
            
            # Calculate aspect ratios
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            mar = mouth_aspect_ratio(mouth)
            
            # Draw contours with thicker lines
            cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 255, 0), 2)
            cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 2)
            cv2.drawContours(frame, [cv2.convexHull(mouth)], -1, (0, 255, 0), 2)
            
            # Check for drowsiness (EAR between thresholds)
            if ear < EYE_AR_DROWSY_THRESH and ear > EYE_AR_THRESH:
                cv2.putText(frame, "DROWSY ALERT!", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                last_state = "DROWSY"
            
            # Check for closed eyes (play sound)
            if ear < EYE_AR_THRESH:
                COUNTER += 1
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    if current_time - last_alarm_time >= MIN_ALARM_INTERVAL:
                        play_alarm()
                        last_alarm_time = current_time
                        last_state = "CLOSED EYES"
                    cv2.putText(frame, "EYES CLOSED!", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                COUNTER = 0
                if last_state == "CLOSED EYES":
                    last_state = "AWAKE"
            
            # Check for yawning (play sound)
            if mar > MAR_THRESH:
                YAWN_COUNTER += 1
                if YAWN_COUNTER >= YAWN_CONSEC_FRAMES:
                    if current_time - last_alarm_time >= MIN_ALARM_INTERVAL:
                        play_alarm()
                        last_alarm_time = current_time
                        last_state = "YAWNING"
                    cv2.putText(frame, "YAWN DETECTED!", (10, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                YAWN_COUNTER = 0
                if last_state == "YAWNING":
                    last_state = "AWAKE"
            
            # Display information with better visibility
            cv2.putText(frame, f"EAR: {ear:.2f}", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"MAR: {mar:.2f}", (10, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"State: {last_state}", (10, 180),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show frame
        cv2.imshow("Drowsiness Detection", frame)
        
        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()