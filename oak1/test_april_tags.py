import cv2
import depthai as dai
from pupil_apriltags import Detector

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

# Create AprilTag detector
at_detector = Detector(families='tag36h11',
                       nthreads=1,
                       quad_decimate=2.0,
                       quad_sigma=0.0,
                       refine_edges=1,
                       decode_sharpening=0.25,
                       debug=0)

pipeline.start()
while pipeline.isRunning():
    videoIn = videoQueue.get()  # blocking
    frame = videoIn.getCvFrame()

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect AprilTags
    tags = at_detector.detect(gray)

    for tag in tags:
        # Extract corners for drawing
        (ptA, ptB, ptC, ptD) = tag.corners
        
        # Draw the bounding box
        cv2.line(frame, tuple(ptA.astype(int)), tuple(ptB.astype(int)), (0, 255, 0), 2)
        cv2.line(frame, tuple(ptB.astype(int)), tuple(ptC.astype(int)), (0, 255, 0), 2)
        cv2.line(frame, tuple(ptC.astype(int)), tuple(ptD.astype(int)), (0, 255, 0), 2)
        cv2.line(frame, tuple(ptD.astype(int)), tuple(ptA.astype(int)), (0, 255, 0), 2)

        # Draw the ID
        cv2.putText(frame, str(tag.tag_id), (int(ptA[0]), int(ptA[1] - 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow("video", frame)   
    if cv2.waitKey(1) == ord('q'):
        break