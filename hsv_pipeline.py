import cv2
import numpy as np

def hsv_filter(image, lower_hsv, upper_hsv):
    """
    Applies an HSV color filter to an image.

    Args:
        image (numpy.ndarray): The input image in BGR format.
        lower_hsv (tuple): A tuple (H, S, V) representing the lower bound of the HSV threshold.
        upper_hsv (tuple): A tuple (H, S, V) representing the upper bound of the HSV threshold.

    Returns:
        numpy.ndarray: The masked image after applying the HSV filter.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    res = cv2.bitwise_and(image, image, mask=mask)
    return res, mask

if __name__ == '__main__':
    # Example usage:
    # Create a dummy image for testing
    dummy_image = np.zeros((300, 500, 3), dtype=np.uint8)
    cv2.rectangle(dummy_image, (100, 100), (200, 200), (0, 255, 0), -1) # Green rectangle
    cv2.rectangle(dummy_image, (250, 100), (350, 200), (0, 0, 255), -1) # Red rectangle
    cv2.rectangle(dummy_image, (400, 100), (450, 200), (255, 0, 0), -1) # Blue rectangle

    # Define some HSV bounds (example for green color)
    lower_green = (40, 40, 40)
    upper_green = (80, 255, 255)

    filtered_image, mask = hsv_filter(dummy_image, lower_green, upper_green)

    cv2.imshow("Original Image", dummy_image)
    cv2.imshow("Filtered Image (Green)", filtered_image)
    cv2.imshow("Mask (Green)", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
