import cv2
import numpy as np
# Assuming Threshold and OpenCVPipeline are defined in these imports
from util import Threshold 
# Assuming pipeline_base.py contains the unmodified OpenCVPipeline class
from pipeline_base import OpenCVPipeline 

# --- HSV Filter and Pipeline Classes (Kept from Previous Example) ---

def hsv_filter(image, threshold: Threshold):
    """
    Applies an HSV color filter to an image.
    """
    lower_hsv = threshold.lower_bound
    upper_hsv = threshold.upper_bound

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_hsv_np = np.array(lower_hsv, dtype="uint8")
    upper_hsv_np = np.array(upper_hsv, dtype="uint8")

    mask = cv2.inRange(hsv, lower_hsv_np, upper_hsv_np)
    res = cv2.bitwise_and(image, image, mask=mask)
    return res, mask

class HSVPipeline(OpenCVPipeline):
    """
    A pipeline that applies a color threshold filter based on HSV values.
    """
    def __init__(self, lower_hsv = None, upper_hsv = None, threshold: Threshold = None):
        super().__init__()
        # Use a default threshold if none is provided to avoid errors
        self.threshold = Threshold(lower_hsv or (0,0,0), upper_hsv or (180, 255, 255))
        if threshold is not None: 
            self.threshold = threshold
        self.last_mask = None 

    def process_frame(self, frame):
        """
        Applies the HSV filter to the input frame.
        NOTE: This is the ONLY place where we can read the trackbar values 
              since the base pipeline's run() method calls this function every frame.
        """
        if frame is None:
            return None
        
        # --- TRACKBAR UPDATE LOGIC INJECTED HERE ---
        # Read the current trackbar positions to dynamically update the threshold
        
        # NOTE: This assumes 'HSV Tuning' window has been created in __main__
        try:
            l_h = cv2.getTrackbarPos('Low H', 'HSV Tuning')
            h_h = cv2.getTrackbarPos('High H', 'HSV Tuning')
            l_s = cv2.getTrackbarPos('Low S', 'HSV Tuning')
            h_s = cv2.getTrackbarPos('High S', 'HSV Tuning')
            l_v = cv2.getTrackbarPos('Low V', 'HSV Tuning')
            h_v = cv2.getTrackbarPos('High V', 'HSV Tuning')
            
            # Update the pipeline's threshold object with new values
            self.threshold.lower_bound = (l_h, l_s, l_v)
            self.threshold.upper_bound = (h_h, h_s, h_v)
        except cv2.error:
            # Handle case where trackbar window hasn't been created yet or is destroyed
            pass
        # --- END TRACKBAR UPDATE LOGIC ---
        
        # hsv_filter returns (result_image, mask)
        processed_frame, self.last_mask = hsv_filter(frame, self.threshold)
        
        # Optional: Display the mask alongside the processed frame for better debugging
        if self.last_mask is not None:
             cv2.imshow('Mask (Binary Filter)', self.last_mask)
             
        # The base OpenCVPipeline.run() will display the 'Processed Frame'
        return processed_frame

# --- Real-Time Tuner Setup ---

def nothing(x):
    """A dummy callback function required by cv2.createTrackbar."""
    pass

if __name__ == "__main__":
    # 1. Define initial values for a target color (e.g., Purple)
    initial_lower = (111, 64, 40)
    initial_upper = (177, 139, 80)
    initial_threshold = Threshold(initial_lower, initial_upper)

    # 2. Instantiate the pipeline with initial values
    hsv_pipeline = HSVPipeline(threshold=initial_threshold)
    
    # 3. Setup Trackbar Control Window
    TUNING_WINDOW = 'HSV Tuning'
    cv2.namedWindow(TUNING_WINDOW)

    # Create trackbars for H, S, V lower and upper bounds
    # Hue max is 179 in OpenCV, Saturation/Value max is 255
    cv2.createTrackbar('Low H', TUNING_WINDOW, initial_lower[0], 179, nothing)
    cv2.createTrackbar('High H', TUNING_WINDOW, initial_upper[0], 179, nothing)
    cv2.createTrackbar('Low S', TUNING_WINDOW, initial_lower[1], 255, nothing)
    cv2.createTrackbar('High S', TUNING_WINDOW, initial_upper[1], 255, nothing)
    cv2.createTrackbar('Low V', TUNING_WINDOW, initial_lower[2], 255, nothing)
    cv2.createTrackbar('High V', TUNING_WINDOW, initial_upper[2], 255, nothing)
    
    print("Running HSVPipeline test with real-time tuning using FLIR wrapper...")
    print(f"Initial Threshold: {initial_threshold.lower_bound} to {initial_threshold.upper_bound}")
        
    # 4. CRITICAL STEP: Call the unmodified pipeline.run() with input_source=-1
    # The tuning logic is now *inside* HSVPipeline.process_frame()
    hsv_pipeline.run(input_source=-1) 
    
    # 5. Print the final settings when the loop exits
    print("\n--- Final HSV Settings ---")
    print(f"Lower Bound: {hsv_pipeline.threshold.lower_bound}")
    print(f"Upper Bound: {hsv_pipeline.threshold.upper_bound}")
    print("Real-time tuning finished. All windows destroyed.")