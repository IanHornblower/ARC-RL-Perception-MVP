import PySpin
import cv2
import numpy as np
import sys

class CameraWrapper:
    def __init__(self, camera_index=0):
        self.system = None
        self.cam = None
        self.cam_list = None
        self.camera_index = camera_index
        self.is_initialized = False

    def _configure_pixel_format(self, cam, nodemap):
        """
        Configures the camera's pixel format to a color format if available.
        """
        try:
            node_pixel_format = PySpin.CEnumerationPtr(nodemap.GetNode('PixelFormat'))
            if not PySpin.IsReadable(node_pixel_format) or not PySpin.IsWritable(node_pixel_format):
                print('Unable to get or set pixel format. Aborting...')
                return False

            # Try to set to BGR8
            if PySpin.IsAvailable(node_pixel_format.GetEntryByName('BGR8')):
                node_pixel_format_bgr8 = node_pixel_format.GetEntryByName('BGR8')
                node_pixel_format.SetIntValue(node_pixel_format_bgr8.GetValue())
                print('Pixel format set to BGR8.')
            # If BGR8 is not available, try BayerRG8
            elif PySpin.IsAvailable(node_pixel_format.GetEntryByName('BayerRG8')):
                node_pixel_format_bayerrg8 = node_pixel_format.GetEntryByName('BayerRG8')
                node_pixel_format.SetIntValue(node_pixel_format_bayerrg8.GetValue())
                print('Pixel format set to BayerRG8.')
            else:
                print('No suitable color pixel format (BGR8 or BayerRG8) found. Using current format.')

            current_pixel_format = PySpin.CEnumerationPtr(nodemap.GetNode('PixelFormat')).GetCurrentEntry().GetDisplayName()
            print(f'Current pixel format: {current_pixel_format}')

        except PySpin.SpinnakerException as ex:
            print('Error configuring pixel format: %s' % ex)
            return False
        return True

    def initialize_camera(self):
        """
        Initializes the camera system and the specific camera.
        """
        if self.is_initialized:
            print("Camera already initialized.")
            return True

        self.system = PySpin.System.GetInstance()
        self.cam_list = self.system.GetCameras()
        num_cameras = self.cam_list.GetSize()

        if num_cameras == 0:
            print('No cameras detected.')
            self.cam_list.Clear()
            self.system.ReleaseInstance()
            return False

        if self.camera_index >= num_cameras:
            print(f'Camera index {self.camera_index} out of range. Only {num_cameras} cameras detected.')
            self.cam_list.Clear()
            self.system.ReleaseInstance()
            return False

        self.cam = self.cam_list.GetByIndex(self.camera_index)
        try:
            self.cam.Init()
            nodemap = self.cam.GetNodeMap()
            nodemap_tldevice = self.cam.GetTLDeviceNodeMap()

            # Configure acquisition mode to continuous
            node_acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
            if not PySpin.IsReadable(node_acquisition_mode) or not PySpin.IsWritable(node_acquisition_mode):
                print('Unable to set acquisition mode to continuous (enum retrieval). Aborting...')
                return False

            node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
            if not PySpin.IsReadable(node_acquisition_mode_continuous):
                print('Unable to set acquisition mode to continuous (entry retrieval). Aborting...')
                return False

            acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()
            node_acquisition_mode.SetIntValue(acquisition_mode_continuous)
            print('Acquisition mode set to continuous...')

            # Configure pixel format for color
            if not self._configure_pixel_format(self.cam, nodemap):
                return False

            # Set bufferhandling mode to NewestOnly
            sNodemap = self.cam.GetTLStreamNodeMap()
            node_bufferhandling_mode = PySpin.CEnumerationPtr(sNodemap.GetNode('StreamBufferHandlingMode'))
            if not PySpin.IsReadable(node_bufferhandling_mode) or not PySpin.IsWritable(node_bufferhandling_mode):
                print('Unable to set stream buffer handling mode.. Aborting...')
                return False

            node_newestonly = node_bufferhandling_mode.GetEntryByName('NewestOnly')
            if not PySpin.IsReadable(node_newestonly):
                print('Unable to set stream buffer handling mode.. Aborting...')
                return False

            node_newestonly_mode = node_newestonly.GetValue()
            node_bufferhandling_mode.SetIntValue(node_newestonly_mode)
            print('Stream buffer handling mode set to NewestOnly.')

            self.cam.BeginAcquisition()
            self.is_initialized = True
            print(f'Camera {self.camera_index} initialized and acquisition started.')
            return True
        except PySpin.SpinnakerException as ex:
            print('Error during camera initialization: %s' % ex)
            self.release_camera()
            return False

    def get_frame(self):
        """
        Acquires a single image frame from the camera and converts it to BGR format.
        """
        if not self.is_initialized:
            print("Camera not initialized. Call initialize_camera() first.")
            return None

        try:
            image_result = self.cam.GetNextImage(1000) # 1000 ms timeout

            if image_result.IsIncomplete():
                print('Image incomplete with image status %d ...' % image_result.GetImageStatus())
                return None
            else:
                img_np = image_result.GetNDArray()
                pixel_format = image_result.GetPixelFormat()

                if pixel_format == PySpin.PixelFormat_BayerRG8:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_BayerRG2BGR)
                elif pixel_format == PySpin.PixelFormat_BayerGR8:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_BayerGR2BGR)
                elif pixel_format == PySpin.PixelFormat_BayerGB8:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_BayerGB2BGR)
                elif pixel_format == PySpin.PixelFormat_BayerBG8:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_BayerBG2BGR)
                elif pixel_format == PySpin.PixelFormat_Mono8:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
                elif pixel_format == PySpin.PixelFormat_RGB8:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                # If it's already BGR8, no conversion is needed.

                # Applying a diagnostic R/B channel swap to address reported color issues.
                # If this resolves the problem, the root cause is an R/B channel mismatch.
                img_np = img_np[..., ::-1]

                image_result.Release()
                return img_np

        except PySpin.SpinnakerException as ex:
            print('Error acquiring image: %s' % ex)
            return None

    def release_camera(self):
        """
        Ends acquisition and releases camera resources.
        """
        if self.is_initialized:
            try:
                self.cam.EndAcquisition()
                self.cam.DeInit()
                print(f'Camera {self.camera_index} acquisition ended and deinitialized.')
            except PySpin.SpinnakerException as ex:
                print('Error during camera release: %s' % ex)
            finally:
                self.is_initialized = False
        if self.cam_list is not None:
            del self.cam
            self.cam_list.Clear()
        if self.system is not None:
            self.system.ReleaseInstance()
        cv2.destroyAllWindows()
        print("Camera resources released.")

if __name__ == '__main__':
    # Example usage:
    camera_wrapper = CameraWrapper(camera_index=0)
    if camera_wrapper.initialize_camera():
        print("Press 'q' to quit.")
        while True:
            frame = camera_wrapper.get_frame()
            if frame is not None:
                cv2.imshow('Camera Feed', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        camera_wrapper.release_camera()
    else:
        print("Failed to initialize camera.")
    sys.exit(0)
