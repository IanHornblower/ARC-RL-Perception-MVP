import cv2
import numpy as np
import argparse
from hsv_pipeline import hsv_filter
from flir_camera_wrapper import CameraWrapper

def nothing(x):
    pass

def hsv_tuner(image_path=None, camera_index=0):
    """
    Provides an interactive HSV color tuner with trackbars.

    Args:
        image_path (str, optional): Path to an image file. If provided, the tuner will use this image.
                                    Defaults to None, in which case it uses the camera.
        camera_index (int, optional): Index of the camera to use if no image_path is provided.
                                      Defaults to 0 (default camera).
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
                print("Error: Could not open FLIR camera")
        else:
            cap = cv2.VideoCapture(camera_index)
            if not cap.isOpened():
                print(f"Error: Could not open camera with index {camera_index}")
                return
            original_image = None

    cv2.namedWindow("HSV Tuner")
    cv2.createTrackbar("LH", "HSV Tuner", 0, 255, nothing)
    cv2.createTrackbar("LS", "HSV Tuner", 0, 255, nothing)
    cv2.createTrackbar("LV", "HSV Tuner", 0, 255, nothing)
    cv2.createTrackbar("UH", "HSV Tuner", 255, 255, nothing)
    cv2.createTrackbar("US", "HSV Tuner", 255, 255, nothing)
    cv2.createTrackbar("UV", "HSV Tuner", 255, 255, nothing)

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

        lh = cv2.getTrackbarPos("LH", "HSV Tuner")
        ls = cv2.getTrackbarPos("LS", "HSV Tuner")
        lv = cv2.getTrackbarPos("LV", "HSV Tuner")
        uh = cv2.getTrackbarPos("UH", "HSV Tuner")
        us = cv2.getTrackbarPos("US", "HSV Tuner")
        uv = cv2.getTrackbarPos("UV", "HSV Tuner")

        lower_hsv = (lh, ls, lv)
        upper_hsv = (uh, us, uv)

        filtered_image, mask = hsv_filter(display_image, lower_hsv, upper_hsv)

        cv2.imshow("Original", display_image)
        cv2.imshow("Filtered", filtered_image)
        cv2.imshow("Mask", mask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    if cap:
        cap.release()
    cv2.destroyAllWindows()
    print(f"Final HSV Bounds: Lower={lower_hsv}, Upper={upper_hsv}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Interactive HSV Color Tuner")
    parser.add_argument("--image", type=str, help="Path to an image file for tuning.")
    parser.add_argument("--camera", type=int, default=-1, help="Camera index to use (default: 0).")
    args = parser.parse_args()

    hsv_tuner(image_path=args.image, camera_index=args.camera)
