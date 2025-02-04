import mediapipe as mp
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the saved model
model = load_model("asl_model.h5")

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Label mapping
labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
labels.remove('J')  # Exclude 'J'
labels.remove('Z')  # Exclude 'Z'

# Start webcam feed
cap = cv2.VideoCapture(0)
with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip and convert the frame to RGB
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Hands
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Draw hand landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extract bounding box
                h, w, _ = frame.shape
                x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w)
                y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h)
                x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * w)
                y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * h)

                # Crop and preprocess the hand region
                hand_img = frame[y_min:y_max, x_min:x_max]
                if hand_img.size > 0:
                    hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
                    hand_img = cv2.resize(hand_img, (28, 28)) / 255.0
                    hand_img = hand_img.reshape(1, 28, 28, 1)

                    # Predict the sign
                    predictions = model.predict(hand_img)
                    predicted_class = np.argmax(predictions)
                    label = labels[predicted_class]

                    # Display the label
                    cv2.putText(frame, label, (x_min, y_min - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the frame
        cv2.imshow("Sign Language Detection", frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
