import cv2
import mediapipe as mp
import numpy as np
from extend_line import extend_line

mpHand = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils
hands = mpHand.Hands(min_detection_confidence=0.8)

cap = cv2.VideoCapture(0)
ws, hs = 1280, 720
cap.set(3, ws)
cap.set(4, hs)

while cap.isOpened():
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    multiHandDetection = results.multi_hand_landmarks  # Hand Detection
    lmList = []

    if multiHandDetection:
        # Hand Visualization for index and middle fingers
        for id, lm in enumerate(multiHandDetection[0].landmark):
            if id == 8:  # Middle finger landmark
                h, w, c = img.shape  # Get image shape here
                lm_x7, lm_y7 = int(lm.x * ws), int(lm.y * hs)  # Use ws and hs for width and height
                lmList.append([lm_x7, lm_y7])
                # Draw a bounding box (rectangle) around the middle finger landmark (7)
                box_size = 10  # Set the size of the bounding box
                top_left = (lm_x7 - box_size // 2, lm_y7 - box_size // 2)
                bottom_right = (lm_x7 + box_size // 2, lm_y7 + box_size // 2)
                cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)

            if id == 7:  # Index finger landmark
                lm_x8, lm_y8 = int(lm.x * ws), int(lm.y * hs)  # Use ws and hs for width and height
                lmList.append([lm_x8, lm_y8])

        # Draw a line between landmarks 7 (middle finger) and 8 (index finger)
        if len(lmList) == 2:  # Ensure both landmarks are detected
            lm_x7, lm_y7 = lmList[0]
            lm_x8, lm_y8 = lmList[1]

            # Extend the line between landmarks 7 and 8
            x1_extended, y1_extended, x2_extended, y2_extended = extend_line(lm_x7, lm_y7, lm_x8, lm_y8, length=100)

            # Draw the extended line on the image
            #cv2.line(img, (x1_extended, y1_extended), (x2_extended, y2_extended), (0, 255, 0), )  # Line with extension
            # Place the cropped bounding box at the end of the extended line
            crop_size = 224
            x1_crop = max(x2_extended+30 - crop_size // 2, 0)
            y1_crop = max(y2_extended+30 - crop_size // 2, 0)
            x2_crop = min(x2_extended+30 + crop_size // 2, ws)
            y2_crop = min(y2_extended+30 + crop_size // 2, hs)

            # Ensure the crop region is valid (non-empty)
            if x2_crop > x1_crop and y2_crop > y1_crop:
                # Crop the region from the image
                cropped_img = img[y1_crop:y2_crop, x1_crop:x2_crop]

                # Resize the cropped image to 224x224
                cropped_img_resized = cv2.resize(cropped_img, (224, 224))

                # Display the cropped and resized image on the main image
                cv2.rectangle(img, (x1_crop, y1_crop), (x2_crop, y2_crop), (0, 0, 255), 2)  # bounding box for the crop
                cv2.imshow("cROPPED",cropped_img_resized)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cv2.destroyAllWindows()
