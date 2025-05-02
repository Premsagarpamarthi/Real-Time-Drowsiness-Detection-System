import os
import cv2
import dlib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

# Initialize dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def extract_landmarks(image):
    """Extract facial landmarks from an image"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    if len(faces) == 0:
        return None
    
    landmarks = []
    for face in faces:
        shape = predictor(gray, face)
        for i in range(68):
            x = shape.part(i).x
            y = shape.part(i).y
            landmarks.extend([x, y])
    
    return np.array(landmarks)

def load_data(data_dir):
    """Load and preprocess training data"""
    X = []
    y = []
    class_names = ['awake', 'drowsy', 'yawn']
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            continue
            
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img = cv2.imread(img_path)
            
            if img is None:
                continue
                
            landmarks = extract_landmarks(img)
            if landmarks is not None:
                X.append(landmarks)
                y.append(class_idx)
    
    return np.array(X), np.array(y)

def train_model(X, y):
    """Train the drowsiness detection model"""
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train SVM classifier
    clf = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
    clf.fit(X_train_scaled, y_train)
    
    # Evaluate model
    train_accuracy = clf.score(X_train_scaled, y_train)
    test_accuracy = clf.score(X_test_scaled, y_test)
    
    print(f"Training accuracy: {train_accuracy:.2f}")
    print(f"Testing accuracy: {test_accuracy:.2f}")
    
    return clf, scaler

def main():
    # Load training data
    print("Loading training data...")
    X, y = load_data("training_data")
    
    if len(X) == 0:
        print("No training data found!")
        return
    
    print(f"Loaded {len(X)} samples")
    
    # Train model
    print("Training model...")
    model, scaler = train_model(X, y)
    
    # Save model and scaler
    print("Saving model...")
    joblib.dump(model, 'drowsiness_model.joblib')
    joblib.dump(scaler, 'drowsiness_scaler.joblib')
    print("Model saved successfully!")

if __name__ == "__main__":
    main()

