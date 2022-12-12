import cv2
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('emotion_detector.h5')

# Initialize the camera
camera = cv2.VideoCapture(0)

while True:
    # Capture a frame from the camera
    _, frame = camera.read()
    
    # Preprocess the frame (e.g., resize, convert to grayscale)
    processed_frame = preprocess_frame(frame)

    # Use the model to predict the emotion of the subjects in the frame
    predictions = model.predict(processed_frame)

    # Display the predicted emotions on the frame
    display_emotions(frame, predictions)

    # Show the frame
    cv2.imshow('Emotion Detector', frame)

    # Press q to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Release the camera
camera.release()
