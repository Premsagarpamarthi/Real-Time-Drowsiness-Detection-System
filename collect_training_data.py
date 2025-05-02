import cv2
import os
import time
import dlib
import numpy as np

# Initialize dlib's face detector
detector = dlib.get_frontal_face_detector()

# Create directories if they don't exist
directories = ['training_data/awake', 'training_data/drowsy', 'training_data/yawn']
for directory in directories:
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_next_image_number(directory):
    """Get the next available image number in the directory"""
    existing_files = [f for f in os.listdir(directory) if f.endswith('.jpg')]
    if not existing_files:
        return 1
    return max([int(f.split('.')[0]) for f in existing_files]) + 1

def list_cameras():
    """List all available cameras with their properties"""
    print("\nAvailable cameras:")
    for i in range(5):
        try:
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                backend = cap.getBackendName()
                print(f"\nCamera {i}:")
                print(f"Resolution: {width}x{height}")
                print(f"FPS: {fps}")
                print(f"Backend: {backend}")
                cap.release()
        except:
            continue

def main():
    # List available cameras first
    list_cameras()
    
    # Try to use camera index 1 (often the physical webcam)
    print("\nTrying to use camera index 1 (physical webcam)...")
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print("\nError: Could not open camera index 1")
        print("Please check Device Manager to identify your webcam's index")
        print("1. Press Windows + X and select 'Device Manager'")
        print("2. Expand the 'Cameras' section")
        print("3. Note the index of your physical webcam")
        print("4. Disable any mobile/phone cameras")
        return

    # Dictionary to map keys to categories
    categories = {
        'a': 'awake',
        'd': 'drowsy',
        'y': 'yawn'
    }

    print("\nInstructions:")
    print("Press 'a' to capture awake image")
    print("Press 'd' to capture drowsy image")
    print("Press 'y' to capture yawn image")
    print("Press 'q' to quit")
    print("\nMake sure your face is clearly visible in the frame")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break

        # Detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        # Draw rectangle around face
        for face in faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display instructions
        cv2.putText(frame, "Press 'a' for awake, 'd' for drowsy, 'y' for yawn", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'q' to quit", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Collect Training Data', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key in [ord('a'), ord('d'), ord('y')]:
            category = categories[chr(key)]
            if len(faces) > 0:
                # Save the image
                img_num = get_next_image_number(f'training_data/{category}')
                img_path = f'training_data/{category}/{img_num}.jpg'
                cv2.imwrite(img_path, frame)
                print(f"Saved {category} image: {img_path}")
                # Show confirmation
                cv2.putText(frame, "Image saved!", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow('Collect Training Data', frame)
                cv2.waitKey(1000)  # Show confirmation for 1 second
            else:
                print("No face detected! Please try again.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 