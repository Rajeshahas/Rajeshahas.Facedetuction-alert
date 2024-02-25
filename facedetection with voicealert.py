import cv2
import pyttsx3

# Load the pre-trained face detection classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#bottom code is used for package pasted in project file if not pasted use top line code.
#face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize the pyttsx3 engine
engine = pyttsx3.init()

# Load the video capture
cap = cv2.VideoCapture(0)  # Use 0 for the default camera

while True:
    # Read the frame from the camera
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform face detection
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Check if a face is detected
    if len(faces) > 0:
        # Display alert message
        engine.say('Face detected')
        engine.runAndWait()

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Display the result
    cv2.imshow('Face Detection', frame)

    # Check for the 'q' key to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture
cap.release()
cv2.destroyAllWindows()
