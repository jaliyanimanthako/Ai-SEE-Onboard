import cv2
import mediapipe as mp
import numpy as np
import time
from extend_line import extend_line
from vizualize import visualize

# Function to run object detection on a cropped image
def run_object_detection_on_cropped_image(model_path, cropped_image):
    """Runs object detection on a single cropped image."""
    # Set up MediaPipe object detection
    BaseOptions = mp.tasks.BaseOptions
    ObjectDetector = mp.tasks.vision.ObjectDetector
    ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = ObjectDetectorOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        max_results=5,
        running_mode=VisionRunningMode.IMAGE,  # Single image detection mode
        score_threshold=0.5
    )

    with ObjectDetector.create_from_options(options) as detector:
        # Convert the image from BGR to RGB (MediaPipe uses RGB format)
        cropped_image_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)

        # Create MediaPipe image from the RGB frame
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(cropped_image_rgb))

        # Perform detection
        start_time = time.time()
        detection_result = detector.detect(mp_image)
        print("Detection time: ", time.time() - start_time)

        # Visualize the results on the image
        annotated_image = visualize(cropped_image, detection_result)
        return annotated_image

# Initialize Mediapipe Hands
mpHand = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils
hands = mpHand.Hands(min_detection_confidence=0.8)

# Set up video capture
cap = cv2.VideoCapture(0)
ws, hs = 1280, 720
cap.set(3, ws)
cap.set(4, hs)

# Model path
model_path = "ED.tflite"

while cap.isOpened():
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        break

    # Process the frame
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    lmList = []

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                if id == 8:  # Middle finger landmark
                    lm_x7, lm_y7 = int(lm.x * ws), int(lm.y * hs)
                    lmList.append([lm_x7, lm_y7])
                    box_size = 10
                    top_left = (lm_x7 - box_size // 2, lm_y7 - box_size // 2)
                    bottom_right = (lm_x7 + box_size // 2, lm_y7 + box_size // 2)
                    cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)

                if id == 7:  # Index finger landmark
                    lm_x8, lm_y8 = int(lm.x * ws), int(lm.y * hs)
                    lmList.append([lm_x8, lm_y8])

            # Draw connections and landmarks
            mpDraw.draw_landmarks(img, handLms, mpHand.HAND_CONNECTIONS)

        # Draw a line between landmarks 7 and 8 and crop region
        if len(lmList) == 2:
            lm_x7, lm_y7 = lmList[0]
            lm_x8, lm_y8 = lmList[1]

            # Extend the line between landmarks 7 and 8
            x1_extended, y1_extended, x2_extended, y2_extended = extend_line(lm_x7, lm_y7, lm_x8, lm_y8, length=100)

            # Define the crop region
            crop_size = 224
            x1_crop = max(x2_extended + 30 - crop_size // 2, 0)
            y1_crop = max(y2_extended + 30 - crop_size // 2, 0)
            x2_crop = min(x2_extended + 30 + crop_size // 2, ws)
            y2_crop = min(y2_extended + 30 + crop_size // 2, hs)

            if x2_crop > x1_crop and y2_crop > y1_crop:
                cropped_img = img[y1_crop:y2_crop, x1_crop:x2_crop]
                cropped_img_resized = cv2.resize(cropped_img, (224, 224))

                # Run object detection on the cropped image
                annotated_cropped_img = run_object_detection_on_cropped_image(model_path, cropped_img_resized)

                # Display the annotated cropped image
                cv2.imshow("Object Detection on Cropped Image", annotated_cropped_img)

    # Display the camera feed
    cv2.imshow("Camera Feed", img)

    key = cv2.waitKey(1)
    if key == ord('q'):
        # Exit the loop
        break

# Release resources
cap.release()
hands.close()
cv2.destroyAllWindows()
