import cv2
import numpy as np
# Assuming hsv_filter and Threshold are available from your original imports
from util import Threshold
from hsv_pipeline import hsv_filter
# Assuming OpenCVPipeline is available from your original PIPELINE definition
from pipeline_base import OpenCVPipeline # Assuming PIPELINE is a module name

def get_blob_centroids(blobs):
    """
    Calculates the centroid (center of mass) for a list of contours.
    
    Args:
        blobs (list): A list of OpenCV contours (numpy arrays).
        
    Returns:
        list: A list of (x, y) integer tuples representing the centroids.
    """
    centroids = []
    for c in blobs:
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centroids.append((cX, cY))
    return centroids


class RedBlobFinderPipeline(OpenCVPipeline):
    """
    A pipeline to detect 'red' (or any color defined by the threshold) blobs 
    using HSV filtering and contour detection.
    """
    def __init__(self, threshold: Threshold, min_blob_area = 200):
        """
        Initializes the pipeline with the specific HSV threshold.

        Args:
            threshold (Threshold): An object containing lower and upper HSV bounds.
        """
        super().__init__()
        self.threshold = threshold
        self.min_blob_area = min_blob_area
        self.last_blobs = [] 
    

    def process_frame(self, frame):
        """
        Applies HSV filtering and finds contours (blobs) in a single frame.

        Args:
            frame (numpy.ndarray): The input image frame in BGR format.

        Returns:
            numpy.ndarray: The binary mask, the frame with contours, or the 
                           homography-corrected frame (if 4 blobs found).
        """
        # 1. Apply HSV filter
        # The hsv_filter function returns (filtered_image, mask)
        res, mask = hsv_filter(frame, self.threshold)

        # --- Morphological Operations ---
        erode = 5
        dilate = 5

        erode_element = cv2.getStructuringElement(cv2.MORPH_RECT, (erode, erode))
        dilate_element = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate, dilate))

        # IMPORTANT: Apply morphological operations to the mask, not res!
        current_mask = cv2.dilate(mask, dilate_element)
        # Note: The original code was using 'mask' for the second operation instead of 'current_mask'. 
        # Correcting to use current_mask after dilate for proper closing/opening.
        current_mask = cv2.erode(current_mask, erode_element) 

        # 2. Find Contours
        contours, _ = cv2.findContours(current_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 3. Filter Contours (Blobs)
        self.last_blobs = [c for c in contours if cv2.contourArea(c) > self.min_blob_area]
        
        # print(len(self.last_blobs))

        # --- NEW HOMOGRAPHY LOGIC ---
        if len(self.last_blobs) == 4:
    
            # 4. Homography Calculation
            centroids = get_blob_centroids(self.last_blobs)
            
            # Convert centroids to a numpy array for OpenCV functions
            # Sort the points to ensure consistent order (e.g., top-left, top-right, bottom-right, bottom-left)
            # Simple sorting by Y then X often works for roughly rectangular groups:
            centroids.sort(key=lambda p: (p[1], p[0])) 
            
            # The sorting above might be unreliable. A more robust method is:
            # a) Sort by Y-coordinate to separate Top/Bottom pairs.
            # b) Sort each pair (Top/Bottom) by X-coordinate to get L/R.
            
            # Convert to numpy array of float32
            src_pts = np.array(centroids, dtype=np.float32)

            # Define the destination points (a normalized rectangle)
            # Use the dimensions of the input frame for the warped output size
            h, w = frame.shape[:2]
            
            # Destination points (e.g., a square/rectangle in the center of the frame)
            # We will use the full frame size for simplicity of display, 
            # assuming the 4 points define a perspective-warped view of a flat object.
            
            # A more robust sort is needed before setting dst_pts to ensure consistency:
            # 1. Sort points based on their sum of x+y (Top-Left is min, Bottom-Right is max)
            # 2. Sort remaining two points based on their difference x-y (Top-Right is min, Bottom-Left is max)
            
            # For simplicity, we assume the simple sort by Y then X provides an order that can be 
            # mapped to a standard rectangular order (TL, TR, BR, BL) after some manual reordering if needed.
            
            # Order: Top-Left, Top-Right, Bottom-Right, Bottom-Left
            # If the simple Y,X sort worked, the points should be close to this:
            # [0]=TL, [1]=TR, [2]=BL, [3]=BR (if points are close horizontally)
            
            # A safer approach:
            
            # 1. Determine Top-Left (min sum) and Bottom-Right (max sum)
            s = src_pts.sum(axis=1)
            tl = src_pts[np.argmin(s)]
            br = src_pts[np.argmax(s)]
            
            # 2. Determine Top-Right (min diff) and Bottom-Left (max diff)
            d = np.diff(src_pts, axis=1) # x - y
            tr = src_pts[np.argmin(d)]
            bl = src_pts[np.argmax(d)]
            
            # Reconstruct src_pts in the correct order: TL, TR, BR, BL
            src_pts_sorted = np.array([tl, tr, br, bl], dtype=np.float32)

            # Destination points (e.g., a 600x400 rectangle)
            output_w = 600
            output_h = 400
            dst_pts = np.array([
                [0, 0],
                [output_w - 1, 0],
                [output_w - 1, output_h - 1],
                [0, output_h - 1]], dtype=np.float32)

            
            # Get the homography matrix
            M, mask = cv2.findHomography(src_pts_sorted, dst_pts, cv2.RANSAC, 5.0)

            if M is not None:
                # Apply the homography transformation
                warped_image = cv2.warpPerspective(frame, M, (output_w, output_h))
                return warped_image # Return the resulting warped image
            
        # 5. Default Return (If not 4 blobs or homography failed)
        
        # Draw the found contours on the original frame (for visualization)
        display_frame = frame.copy()
        cv2.drawContours(display_frame, self.last_blobs, -1, (0, 255, 0), 2) # Draw green contours

        # Return the frame with drawn contours
        return display_frame
    
    # Optional utility method to retrieve the last detected blobs
    def get_last_blobs(self):
        return self.last_blobs

# Example Usage:
if __name__ == '__main__':
    # 1. Define a dummy Threshold for red
    # For a color like red, which wraps around the 179 H limit, two ranges are usually needed.
    # The hsv_filter should handle the two ranges (0-10 and 160-179). 
    # Assuming the current hsv_filter function handles this if passed one of the ranges, 
    # or that the 'red' target in the scene is within one continuous range.
    lower_red1=(0,90,160)
    upper_red1=(20,255,255)
    
    lower_purple = (111, 64, 40)
    upper_purple = (177, 139, 80)

    # We will use the first red range. If red objects are not detected, 
    # the hsv_filter or Threshold definition may need adjustment to handle the wrap-around.
    red_threshold = Threshold(lower_red1, upper_red1)

    purple_threshold = Threshold(lower_purple, upper_purple)

    # 2. Instantiate the pipeline
    # Increased min_blob_area slightly, but keeping it low for the camera test
    blob_pipeline = RedBlobFinderPipeline(red_threshold, min_blob_area=2) 

    # # 3. Example with a dummy image (Testing 4 blobs scenario)
    # dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
    # # Draw 4 red squares (BGR: 0, 0, 255) -> HSV approx: 0, 255, 255 (within the threshold)
    # cv2.rectangle(dummy_image, (50, 50), (100, 100), (0, 0, 255), -1) # TL
    # cv2.rectangle(dummy_image, (500, 70), (550, 120), (0, 0, 255), -1) # TR
    # cv2.rectangle(dummy_image, (100, 350), (150, 400), (0, 0, 255), -1) # BL
    # cv2.rectangle(dummy_image, (450, 300), (500, 350), (0, 0, 255), -1) # BR
    
    # # Process the static image directly
    # processed_frame_static = blob_pipeline.process_frame(dummy_image)
    
    # print(f"Detected {len(blob_pipeline.get_last_blobs())} blobs in static image.")
    # if len(blob_pipeline.get_last_blobs()) == 4:
    #     cv2.imshow("Processed Frame (Homography Corrected)", processed_frame_static)
    # else:
    #     cv2.imshow("Processed Frame (Contours)", processed_frame_static)

    # cv2.imshow("Original Image", dummy_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 4. Example usage with camera using the inherited run method (Uncomment to test live)
    # print("\nRunning Red Blob Finder Pipeline (press 'q' to quit)...")
    blob_pipeline.run(input_source=-1)