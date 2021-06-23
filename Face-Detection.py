import cv2

# Load the cascade
face_cascade = cv2.CascadeClassifier('C:/xampp/htdocs/Python Virtual Try On/cascades/haarcascade_frontalface_default.xml')

# To capture video from webcam. 
capture_video = cv2.VideoCapture(0)


while True:
    # Read the frame
    _, img = capture_video.read()

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Display
    cv2.imshow('Face Detection', img)

    # Stop if escape key is pressed
    key = cv2.waitKey(5)
    
    if key ==27:
        break;
        
# Release the VideoCapture object
capture_video.release()