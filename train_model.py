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
    print("[INFO] Starting data collection...")
    print(f"[INFO] Using camera index: {camera_index}")
    
    # Create directories if they don't exist
    # Create directories if they don't exist
    for subdir in [output_dir, 
                   os.path.join(output_dir, "awake"),
                   os.path.join(output_dir, "drowsy"),
                   os.path.join(output_dir, "yawn")]:
        if not os.path.exists(subdir):
            os.makedirs(subdir)
    
    # Initialize face detector and landmark predictor
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    
    # Initialize video capture
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam")
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
        
        # Display instructions
        cv2.putText(frame, "Press 'a' for awake, 'd' for drowsy, 'y' for yawn", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, f"Awake: {awake_count}/{num_samples}", 
                   (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, f"Drowsy: {drowsy_count}/{num_samples}", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, f"Yawn: {yawn_count}/{num_samples}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
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
            
        elif key == ord('q'):
            break
            
        if awake_count >= num_samples and drowsy_count >= num_samples and yawn_count >= num_samples:
            print("[INFO] Data collection complete!")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return True

def extract_features(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return [0.3, 10]  # Default values
        
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    
    faces = detector.detectMultiScale(gray, scaleFactor=1.1,
                                     minNeighbors=5, minSize=(30, 30),
                                     flags=cv2.CASCADE_SCALE_IMAGE)
    
    for (x, y, w, h) in faces:
        rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        # Extract eye aspect ratio
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        
        # Extract mouth features
        top_lip = shape[50:53]
        top_lip = np.concatenate((top_lip, shape[61:64]))
        low_lip = shape[56:59]
        low_lip = np.concatenate((low_lip, shape[65:68]))
        top_mean = np.mean(top_lip, axis=0)
        low_mean = np.mean(low_lip, axis=0)
        mouth_dist = abs(top_mean[1] - low_mean[1])
        
        return [ear, mouth_dist]
    
    return [0.3, 10]  # Default values if no face detected

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def train_model(data_dir, model_output="drowsiness_model.pkl"):
    print("[INFO] Extracting features and training model...")
    
    features = []
    labels = []
    
    # Process training data
    for class_name in ["awake", "drowsy", "yawn"]:
        class_dir = os.path.join(data_dir, class_name)
        if os.path.exists(class_dir):
            for image_file in os.listdir(class_dir):
                if image_file.endswith(".jpg"):
                    image_path = os.path.join(class_dir, image_file)
                    feature = extract_features(image_path)
                    features.append(feature)
                    labels.append(class_name)
    
    if not features:
        print("[ERROR] No training data found")
        return False
    
    # Convert to numpy arrays
    features = np.array(features)
    labels = np.array(labels)
    
    # Split data and train model
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42)
    
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"[INFO] Model accuracy: {accuracy:.2f}")
    
    # Save model
    with open(model_output, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"[INFO] Model saved to {model_output}")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", type=str, default="training_data",
                        help="path to training data directory")
    parser.add_argument("-m", "--model", type=str, default="drowsiness_model.pkl",
                        help="path to output model")
    parser.add_argument("-n", "--num-samples", type=int, default=30,
                        help="number of samples to collect for each class")
    args = vars(parser.parse_args())
    
    # Collect data
    if collect_data(args["data"], args["num_samples"]):
        # Train model
        train_model(args["data"], args["model"])