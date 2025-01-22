import time
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from vizualize import visualize


MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red

def run_object_detection(model_path, camera_index=0):
    """Runs object detection and displays live video with bounding boxes."""
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Set up MediaPipe object detection
    BaseOptions = mp.tasks.BaseOptions
    ObjectDetector = mp.tasks.vision.ObjectDetector
    ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = ObjectDetectorOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        max_results=5,
        running_mode=VisionRunningMode.VIDEO,
        score_threshold=0.5
    )

    with ObjectDetector.create_from_options(options) as detector:
        frame_index = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to grab frame.")
                break

            # Convert the frame from BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Create MediaPipe image from the RGB frame
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(rgb_frame))

            # Perform detection
            start_time = time.time()
            detection_result = detector.detect_for_video(mp_image, frame_index)
            frame_index += 1
            print("Detection time: ", time.time() - start_time)

            # Visualize the results on the image
            annotated_image = visualize(frame, detection_result)

            # Display the frame with bounding boxes
            cv2.imshow("Object Detection", annotated_image)

            # Exit if the user presses the 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
