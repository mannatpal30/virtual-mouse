import cv2
import numpy as np
import pyautogui
import mediapipe as mp
import math

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

x_prev, y_prev = pyautogui.size().width // 2, pyautogui.size().height // 2

def calculate_distance(point1, point2):
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get the position of the index finger tip and thumb tip
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                
                x1 = int(index_finger_tip.x * frame.shape[1])
                y1 = int(index_finger_tip.y * frame.shape[0])
                x_thumb = int(thumb_tip.x * frame.shape[1])
                y_thumb = int(thumb_tip.y * frame.shape[0])

                cv2.circle(frame, (x1, y1), 10, (0, 255, 0), -1)
                cv2.circle(frame, (x_thumb, y_thumb), 10, (255, 0, 0), -1)

                # Calculating the distance between thumb and index finger tips
                distance = calculate_distance(index_finger_tip, thumb_tip)

                # Simulating mouse movement
                x_prev = int(x_prev * 0.8 + x1 * 0.2)
                y_prev = int(y_prev * 0.8 + y1 * 0.2)
                pyautogui.moveTo(x_prev * (pyautogui.size().width / frame.shape[1]), y_prev * (pyautogui.size().height / frame.shape[0]))

                # If the distance is below a certain threshold, simulate a click
                if distance < 0.05:  # Adjust this threshold as needed
                    pyautogui.click()

        cv2.imshow('Virtual Mouse', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
