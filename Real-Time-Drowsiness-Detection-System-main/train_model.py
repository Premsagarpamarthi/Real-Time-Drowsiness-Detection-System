import os
import cv2
import dlib
import numpy as np
import argparse
import pickle
from imutils import face_utils
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from scipy.spatial import distance as dist

def collect_data(output_dir, num_samples=100, camera_index=0):
    """
    Collect training data for drowsiness detection
    """
    print("[INFO] Starting data collection...")
    print(f"[INFO] Using camera index: {camera_index}")
    
    # Create directories if they don't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        os.makedirs(os.path.join(output_dir, "awake"))
        os.makedirs(os.path.join(output_dir, "drowsy"))
        os.makedirs(os.path.join(output_dir, "yawn"))
    
    # Initialize face detector and landmark predictor
    cascade_path = os.path.join(os.path.dirname(__file__), "haarcascade_frontalface_default.xml")
    predictor_path = os.path.join(os.path.dirname(__file__), "shape_predictor_68_face_landmarks.dat")
    
    # Check if files exist, if not, use absolute paths
    if not os.path.exists(cascade_path):
        cascade_path = "C:/Users/tejas/Downloads/Real-Time-Drowsiness-Detection-System-main/Real-Time-Drowsiness-Detection-System-main/haarcascade_frontalface_default.xml"
    
    if not os.path.exists(predictor_path):
        predictor_path = "C:/Users/tejas/Downloads/Real-Time-Drowsiness-Detection-System-main/Real-Time-Drowsiness-Detection-System-main/shape_predictor_68_face_landmarks.dat"
        
    detector = cv2.CascadeClassifier(cascade_path)
    predictor = dlib.shape_predictor(predictor_path)
    
    # Try different backends for camera capture
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    cap = None
    
    for backend in backends:
        try:
            print(f"[INFO] Trying camera backend: {backend}")
            cap = cv2.VideoCapture(camera_index, backend)
            if cap is None or not cap.isOpened():
                print(f"[WARN] Failed to open camera with backend {backend}")
                continue
                
            # Read a test frame to confirm camera works
            ret, test_frame = cap.read()
            if ret and test_frame is not None:
                print(f"[INFO] Successfully connected to camera with backend {backend}")
                break
            else:
                print(f"[WARN] Camera opened but couldn't read frame with backend {backend}")
                cap.release()
        except Exception as e:
            print(f"[ERROR] Exception when trying backend {backend}: {str(e)}")
            if cap is not None:
                cap.release()
    
    if cap is None or not cap.isOpened():
        print("[ERROR] Could not open webcam. Please check your camera connection and permissions.")
        return False
    
    # Initialize counters
    awake_count = 0
    drowsy_count = 0
    yawn_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.resize(frame, (450, 300))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Display instructions (always show these)
        cv2.putText(frame, "Press 'a' for awake, 'd' for drowsy, 'y' for yawn", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, f"Awake: {awake_count}/{num_samples}", 
                   (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, f"Drowsy: {drowsy_count}/{num_samples}", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, f"Yawn: {yawn_count}/{num_samples}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'q' or ESC to quit", 
                   (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Detect faces
        faces = detector.detectMultiScale(gray, scaleFactor=1.1,
                                         minNeighbors=5, minSize=(30, 30),
                                         flags=cv2.CASCADE_SCALE_IMAGE)
        
        # Initialize variables to display
        ear = 0
        mouth_distance = 0
        face_detected = False
        
        for (x, y, w, h) in faces:
            face_detected = True
            rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
            
            # Get facial landmarks
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            
            # Extract eye aspect ratio
            (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
            (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            
            # Calculate eye aspect ratio
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            
            # Extract mouth features for yawn detection
            top_lip = shape[50:53]
            top_lip = np.concatenate((top_lip, shape[61:64]))
            low_lip = shape[56:59]
            low_lip = np.concatenate((low_lip, shape[65:68]))
            top_mean = np.mean(top_lip, axis=0)
            low_mean = np.mean(low_lip, axis=0)
            mouth_distance = abs(top_mean[1] - low_mean[1])
            
            # Draw face rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Draw eyes
            for (ex, ey) in leftEye:
                cv2.circle(frame, (ex, ey), 1, (0, 0, 255), -1)
            for (ex, ey) in rightEye:
                cv2.circle(frame, (ex, ey), 1, (0, 0, 255), -1)
                
            # Draw mouth
            for (mx, my) in np.vstack([top_lip, low_lip]):
                cv2.circle(frame, (mx, my), 1, (255, 0, 0), -1)
        
        # Display metrics
        if face_detected:
            # Determine state based on metrics
            state = "Awake"
            color = (0, 255, 0)  # Green
            
            if ear < 0.25:  # Threshold for drowsiness
                state = "Drowsy"
                color = (0, 0, 255)  # Red
                
            if mouth_distance > 25:  # Threshold for yawn
                state = "Yawning"
                color = (255, 0, 0)  # Blue
                
            # Display metrics and state
            cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(frame, f"Mouth: {mouth_distance:.2f}", (300, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(frame, f"State: {state}", (300, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        else:
            # No face detected
            cv2.putText(frame, "No face detected", (300, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        cv2.imshow("Data Collection", frame)
        key = cv2.waitKey(1) & 0xFF
        
        # Save data based on key press
        if key == ord('a') and awake_count < num_samples:
            filename = os.path.join(output_dir, "awake", f"awake_{awake_count}.jpg")
            cv2.imwrite(filename, frame)
            awake_count += 1
            print(f"[INFO] Saved awake sample {awake_count}/{num_samples}")
            
        elif key == ord('d') and drowsy_count < num_samples:
            filename = os.path.join(output_dir, "drowsy", f"drowsy_{drowsy_count}.jpg")
            cv2.imwrite(filename, frame)
            drowsy_count += 1
            print(f"[INFO] Saved drowsy sample {drowsy_count}/{num_samples}")
            
        elif key == ord('y') and yawn_count < num_samples:
            filename = os.path.join(output_dir, "yawn", f"yawn_{yawn_count}.jpg")
            cv2.imwrite(filename, frame)
            yawn_count += 1
            print(f"[INFO] Saved yawn sample {yawn_count}/{num_samples}")
            
        # Check for exit keys - both 'q' and ESC (27)
        elif key == ord('q') or key == 27:
            print("[INFO] Exiting data collection...")
            break
            
        # Check if we have collected enough samples
        if awake_count >= num_samples and drowsy_count >= num_samples and yawn_count >= num_samples:
            print("[INFO] Data collection complete!")
            break
    
    # Properly release resources
    cap.release()
    cv2.destroyAllWindows()
    
    # Force close all OpenCV windows
    for i in range(1, 5):
        cv2.waitKey(1)
    
    return True

def eye_aspect_ratio(eye):
    """
    Calculate eye aspect ratio
    """
    # Compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    
    # Compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])
    
    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    
    return ear

def extract_features(image_path):
    """
    Extract features from an image for training
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not read image {image_path}")
        return [0.3, 10]  # Default values
        
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Initialize detector and predictor
    cascade_path = os.path.join(os.path.dirname(__file__), "haarcascade_frontalface_default.xml")
    predictor_path = os.path.join(os.path.dirname(__file__), "shape_predictor_68_face_landmarks.dat")
    
    # Check if files exist, if not, use absolute paths
    if not os.path.exists(cascade_path):
        cascade_path = "C:/Users/tejas/Downloads/Real-Time-Drowsiness-Detection-System-main/Real-Time-Drowsiness-Detection-System-main/haarcascade_frontalface_default.xml"
    
    if not os.path.exists(predictor_path):
        predictor_path = "C:/Users/tejas/Downloads/Real-Time-Drowsiness-Detection-System-main/Real-Time-Drowsiness-Detection-System-main/shape_predictor_68_face_landmarks.dat"
        
    detector = cv2.CascadeClassifier(cascade_path)
    predictor = dlib.shape_predictor(predictor_path)
    
    # Detect faces
    faces = detector.detectMultiScale(gray, scaleFactor=1.1,
                                     minNeighbors=5, minSize=(30, 30),
                                     flags=cv2.CASCADE_SCALE_IMAGE)
    
    for (x, y, w, h) in faces:
        rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        
        # Get facial landmarks
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        # Extract eye aspect ratio
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        
        # Calculate eye aspect ratio
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        
        # Extract mouth features for yawn detection
        top_lip = shape[50:53]
        top_lip = np.concatenate((top_lip, shape[61:64]))
        low_lip = shape[56:59]
        low_lip = np.concatenate((low_lip, shape[65:68]))
        top_mean = np.mean(top_lip, axis=0)
        low_mean = np.mean(low_lip, axis=0)
        mouth_distance = abs(top_mean[1] - low_mean[1])
        
        # Create feature vector
        features = [ear, mouth_distance]
        
        return features
    
    # If no face detected, return default features
    return [0.3, 10]  # Default values

def train_model(data_dir, model_output):
    """
    Train a model for drowsiness detection
    """
    print("[INFO] Extracting features and training model...")
    
    # Lists to store features and labels
    features = []
    labels = []
    
    # Process awake samples
    awake_dir = os.path.join(data_dir, "awake")
    if os.path.exists(awake_dir):
        for image_file in os.listdir(awake_dir):
            if image_file.endswith(".jpg"):
                image_path = os.path.join(awake_dir, image_file)
                feature = extract_features(image_path)
                features.append(feature)
                labels.append("awake")
    
    # Process drowsy samples
    drowsy_dir = os.path.join(data_dir, "drowsy")
    if os.path.exists(drowsy_dir):
        for image_file in os.listdir(drowsy_dir):
            if image_file.endswith(".jpg"):
                image_path = os.path.join(drowsy_dir, image_file)
                feature = extract_features(image_path)
                features.append(feature)
                labels.append("drowsy")
    
    # Process yawn samples
    yawn_dir = os.path.join(data_dir, "yawn")
    if os.path.exists(yawn_dir):
        for image_file in os.listdir(yawn_dir):
            if image_file.endswith(".jpg"):
                image_path = os.path.join(yawn_dir, image_file)
                feature = extract_features(image_path)
                features.append(feature)
                labels.append("yawn")
    
    if not features:
        print("[ERROR] No training data found. Please collect data first.")
        return False
        
    # Convert to numpy arrays
    features = np.array(features)
    labels = np.array(labels)
    
    print(f"[INFO] Total samples: {len(features)}")
    print(f"[INFO] Class distribution: {np.unique(labels, return_counts=True)}")
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42)
    
    # Train SVM model
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"[INFO] Model accuracy: {accuracy:.2f}")
    print("[INFO] Classification report:")
    print(report)
    
    # Save the model
    with open(model_output, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"[INFO] Model saved to {model_output}")
    return True

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--data", type=str, default="training_data",
                    help="path to training data directory")
    ap.add_argument("-m", "--model", type=str, default="drowsiness_model.pkl",
                    help="path to output model")
    ap.add_argument("-n", "--num_samples", type=int, default=100,
                    help="number of samples to collect for each class")
    ap.add_argument("-c", "--collect", action="store_true",
                    help="flag to collect training data")
    ap.add_argument("-t", "--train", action="store_true",
                    help="flag to train the model")
    ap.add_argument("-w", "--webcam", type=int, default=0,
                    help="index of webcam to use (default is 0)")
    args = vars(ap.parse_args())
    
    # Collect data if requested
    if args["collect"]:
        collect_data(args["data"], args["num_samples"], args["webcam"])
    
    # Train model if requested
    if args["train"]:
        train_model(args["data"], args["model"])