import cv2
import mediapipe as mp
import numpy as np
import time
from vizualize import visualize

def run_object_detection(model_path, image):
    """Runs object detection on a single image and displays the result with bounding boxes."""
    # Load the image


    # Set up MediaPipe object detection
    BaseOptions = mp.tasks.BaseOptions
    ObjectDetector = mp.tasks.vision.ObjectDetector
    ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = ObjectDetectorOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        max_results=5,
        running_mode=VisionRunningMode.IMAGE,  # Change to IMAGE mode for single image detection
        score_threshold=0.5
    )

    with ObjectDetector.create_from_options(options) as detector:
        # Convert the image from BGR to RGB (MediaPipe uses RGB format)


        # Create MediaPipe image from the RGB frame
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(image))

        # Perform detection
        start_time = time.time()
        detection_result = detector.detect(mp_image)
        print("Detection time: ", time.time() - start_time)

        # Visualize the results on the image
        annotated_image = visualize(image, detection_result)

        # Display the image with bounding boxes
        cv2.imshow("Object Detection", annotated_image)

        # Wait for user input before closing
        cv2.waitKey(0)
        cv2.destroyAllWindows()


