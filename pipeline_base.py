import cv2
from flir_camera_wrapper import CameraWrapper

class OpenCVPipeline:
    """
    Base class for defining an OpenCV image processing pipeline.
    """
    def __init__(self):
        """
        Initializes the pipeline. Subclasses can override this for custom setup.
        """
        pass

    def process_frame(self, frame):
        """
        Abstract method to be implemented by subclasses for processing a single frame.
        
        Args:
            frame (numpy.ndarray): The input image frame.
            
        Returns:
            numpy.ndarray: The processed image frame.
        """
        raise NotImplementedError("Subclasses must implement 'process_frame' method.")

    def run(self, input_source=0):
        if input_source == -1:
            camera_wrapper = CameraWrapper(camera_index=0)

            if not camera_wrapper.initialize_camera():
                print("Failed to initialize camera.")
        else:
            cap = cv2.VideoCapture(input_source)
            if not cap.isOpened():
                print(f"Error: Could not open camera with index {input_source}")
                return

        while True:
            if input_source == -1:
                display_image = camera_wrapper.get_frame()
            elif cap:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame from camera.")
                display_image = frame

            processed_frame = self.process_frame(display_image)
            cv2.imshow("Processed Frame", processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if input_source != -1:
            cap.release()
        cv2.destroyAllWindows()

# Example of a subclass implementing a simple grayscale conversion pipeline
class GrayscalePipeline(OpenCVPipeline):
    def process_frame(self, frame):
        """
        Converts the input frame to grayscale.
        """
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return gray_frame

# Example of a subclass implementing a blur pipeline
class BlurPipeline(OpenCVPipeline):
    def __init__(self, kernel_size=(5, 5)):
        super().__init__()
        self.kernel_size = kernel_size

    def process_frame(self, frame):
        """
        Applies a Gaussian blur to the input frame.
        """
        blurred_frame = cv2.GaussianBlur(frame, self.kernel_size, 0)
        return blurred_frame

# How to use the pipelines:
if __name__ == "__main__":
    print("Running Grayscale Pipeline (press 'q' to quit)...")
    grayscale_pipeline = GrayscalePipeline()
    grayscale_pipeline.run(input_source=-1) # Uncomment to run with webcam

    print("\nRunning Blur Pipeline (press 'q' to quit)...")
    blur_pipeline = BlurPipeline(kernel_size=(7, 7))
    # blur_pipeline.run(input_source=-1) # Uncomment to run with webcam