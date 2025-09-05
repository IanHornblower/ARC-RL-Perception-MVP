import cv2
import numpy as np
import argparse
from ball_detection_pipeline import BallDetector
from flir_camera_wrapper import CameraWrapper

def nothing(x):
    pass

def ball_detector_app(image_path=None, camera_index=0, roi_normalized=None):
    """
    Runs the ball detection pipeline using the BallDetector class with adjustable parameters via trackbars.

    Args:
        image_path (str, optional): Path to an image file. If provided, the app will use this image.
                                    Defaults to None, in which case it uses the camera.
        camera_index (int, optional): Index of the camera to use if no image_path is provided.
                                      Defaults to 0 (default camera).
        roi_normalized (tuple, optional): Normalized ROI (x_min, y_min, x_max, y_max) where values are 0-1.
                                          If None, the entire frame is processed.
    """

    if image_path:
        original_image = cv2.imread(image_path)
        if original_image is None:
            print(f"Error: Could not load image from {image_path}")
            return
        cap = None
    else:
        if camera_index == -1:
            camera_wrapper = CameraWrapper(camera_index=0)

            if not camera_wrapper.initialize_camera():
                print("Failed to initialize camera.")
        else:
            cap = cv2.VideoCapture(camera_index)
            if not cap.isOpened():
                print(f"Error: Could not open camera with index {camera_index}")
                return
            original_image = None

    detector = BallDetector(
        lower_hsv=(42, 38, 19),
        upper_hsv=(102, 194, 255),
        blur_size=3,
        erode_size=8,
        dilate_size=8,
        morph_operation_type="OPENING",
        contour_mode="EXTERNAL_ONLY",
        draw_contours=True,
        min_area=100,
        max_area=50000,
        min_circularity=0.5,
        roi_normalized=roi_normalized
    )

    # Create a window for trackbars
    cv2.namedWindow("Ball Detector Tuner", cv2.WINDOW_NORMAL)
    cv2.createTrackbar("LH", "Ball Detector Tuner", detector.lower_hsv[0], 179, nothing)
    cv2.createTrackbar("LS", "Ball Detector Tuner", detector.lower_hsv[1], 255, nothing)
    cv2.createTrackbar("LV", "Ball Detector Tuner", detector.lower_hsv[2], 255, nothing)
    cv2.createTrackbar("UH", "Ball Detector Tuner", detector.upper_hsv[0], 179, nothing)
    cv2.createTrackbar("US", "Ball Detector Tuner", detector.upper_hsv[1], 255, nothing)
    cv2.createTrackbar("UV", "Ball Detector Tuner", detector.upper_hsv[2], 255, nothing)
    cv2.createTrackbar("Blur Size", "Ball Detector Tuner", detector.blur_size, 10, nothing)
    cv2.createTrackbar("Erode Size", "Ball Detector Tuner", detector.erode_size, 10, nothing)
    cv2.createTrackbar("Dilate Size", "Ball Detector Tuner", detector.dilate_size, 10, nothing)
    cv2.createTrackbar("Min Area", "Ball Detector Tuner", detector.min_area, 10000, nothing)
    cv2.createTrackbar("Max Area", "Ball Detector Tuner", detector.max_area, 1000000, nothing)
    cv2.createTrackbar("Min Circularity", "Ball Detector Tuner", int(detector.min_circularity * 100), 100, nothing) # Scaled by 100 for trackbar

    # Initialize BallDetector with default values


    # Create display windows and make them resizable
    cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
    cv2.namedWindow("HSV Filtered", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Eroded Mask", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Dilated Mask", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Final Mask", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Processed Frame with Blobs", cv2.WINDOW_NORMAL)

    while True:
        if camera_index == -1:
            display_image = camera_wrapper.get_frame()
        elif cap:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame from camera.")
                break
            display_image = frame
        else:
            display_image = original_image.copy()

        # Read trackbar positions
        lh = cv2.getTrackbarPos("LH", "Ball Detector Tuner")
        ls = cv2.getTrackbarPos("LS", "Ball Detector Tuner")
        lv = cv2.getTrackbarPos("LV", "Ball Detector Tuner")
        uh = cv2.getTrackbarPos("UH", "Ball Detector Tuner")
        us = cv2.getTrackbarPos("US", "Ball Detector Tuner")
        uv = cv2.getTrackbarPos("UV", "Ball Detector Tuner")
        erode_size = cv2.getTrackbarPos("Erode Size", "Ball Detector Tuner")
        dilate_size = cv2.getTrackbarPos("Dilate Size", "Ball Detector Tuner")
        min_area = cv2.getTrackbarPos("Min Area", "Ball Detector Tuner")
        max_area = cv2.getTrackbarPos("Max Area", "Ball Detector Tuner")
        min_circularity = cv2.getTrackbarPos("Min Circularity", "Ball Detector Tuner") / 100.0

        # Update detector parameters
        detector.lower_hsv = (lh, ls, lv)
        detector.upper_hsv = (uh, us, uv)
        detector.erode_size = erode_size
        detector.dilate_size = dilate_size
        detector.min_area = min_area
        detector.max_area = max_area
        detector.min_circularity = min_circularity

        processed_frame, detected_blobs, full_mask, filtered_image, eroded_mask_display, dilated_mask_display = detector.process_frame(display_image)

        cv2.imshow("Original", display_image)
        cv2.imshow("HSV Filtered", filtered_image)
        cv2.imshow("Eroded Mask", eroded_mask_display)
        cv2.imshow("Dilated Mask", dilated_mask_display)
        cv2.imshow("Final Mask", full_mask)
        cv2.imshow("Processed Frame with Blobs", processed_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    if cap:
        cap.release()
    cv2.destroyAllWindows()
    print(f"Ball detection application finished.")
    print(f"Final HSV Bounds: Lower={detector.lower_hsv}, Upper={detector.upper_hsv}")
    print(f"Final Morph Ops: Erode={detector.erode_size}, Dilate={detector.dilate_size}")
    print(f"Final Filters: Min Area={detector.min_area}, Max Area={detector.max_area}, Min Circularity={detector.min_circularity}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Ball Detection Application")
    parser.add_argument("--image", type=str, help="Path to an image file for detection.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index to use (default: 0).")
    parser.add_argument("--roi", type=float, nargs=4, metavar=('X_MIN', 'Y_MIN', 'X_MAX', 'Y_MAX'),
                        help="Normalized ROI coordinates (x_min, y_min, x_max, y_max) where values are 0-1.")
    args = parser.parse_args()

    ball_detector_app(image_path=args.image, camera_index=-1, roi_normalized=tuple(args.roi) if args.roi else None)
