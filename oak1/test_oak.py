import cv2
import depthai as dai

# Create pipeline
pipeline = dai.Pipeline()

# Define source and output
camRgb = pipeline.create(dai.node.ColorCamera)

# Properties
camRgb.setPreviewSize(300, 300)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
outputQueue = camRgb.preview.createOutputQueue()

# Linking
videoQueue = camRgb.video.createOutputQueue()

pipeline.start()
while pipeline.isRunning():
    videoIn = videoQueue.get()  # blocking
    cv2.imshow("video", videoIn.getCvFrame())   
    if cv2.waitKey(1) == ord('q'):
        break