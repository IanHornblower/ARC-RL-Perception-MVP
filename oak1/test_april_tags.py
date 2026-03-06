import cv2
import pprint
import depthai as dai
import numpy as np
from pupil_apriltags import Detector, Detection

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
                       nthreads=8,
                       quad_decimate=1.0,
                       quad_sigma=0.0,
                       refine_edges=1,
                       decode_sharpening=0.25,
                       debug=0)

pipeline.start()
while pipeline.isRunning():
    videoIn = videoQueue.get()  # blocking
    frame = videoIn.getCvFrame() # type: ignore
    output = frame.copy()
    warped_image = None


    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect AprilTags
    tags: list[Detection] = at_detector.detect(gray)
    homography_tags: list[Detection | None] = [None] * 4

    for tag in tags:
        # Extract corners for drawing
        (ptA, ptB, ptC, ptD) = tag.corners
        
        
        # Draw the bounding box
        cv2.line(output, tuple(ptA.astype(int)), tuple(ptB.astype(int)), (0, 255, 0), 2)
        cv2.line(output, tuple(ptB.astype(int)), tuple(ptC.astype(int)), (0, 255, 0), 2)
        cv2.line(output, tuple(ptC.astype(int)), tuple(ptD.astype(int)), (0, 255, 0), 2)
        cv2.line(output, tuple(ptD.astype(int)), tuple(ptA.astype(int)), (0, 255, 0), 2)

        # Draw the ID
        cv2.putText(output, str(tag.tag_id), (int(ptA[0]), int(ptA[1] - 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Get april tag centers
        # (tag id, center)

        homography_tags[tag.tag_id] = tag
        
  

        # get first 4 crentroids for homography (when there are atleast 4 points)
        if  not any(x is None for x in homography_tags):
            top_right: Detection = homography_tags[0]
            bottom_right = homography_tags[1]
            bottom_left = homography_tags[2]
            top_left = homography_tags[3]


            (_, _, _, top_right_corner) = top_right.corners # works 
            (bottom_right_corner, _, _, _) = bottom_right.corners # works 
            (_, _, _, bottom_left_corner) = bottom_left.corners # works 
            (top_left_corner, _, _, _) = top_left.corners # works 
    
            
            cv2.circle(output, tuple(top_right_corner.astype(int)), 2, (0, 0, 255), -1)
            cv2.circle(output, tuple(bottom_right_corner.astype(int)), 2, (0, 0, 255), -1)
            cv2.circle(output, tuple(bottom_left_corner.astype(int)), 2, (0, 0, 255), -1)
            cv2.circle(output, tuple(top_left_corner.astype(int)), 2, (0, 0, 255), -1)

            src_pts = np.array([
                top_right_corner, 
                bottom_right_corner,
                bottom_left_corner,
                top_left_corner
                ])

            output_w = 400
            output_h = 400
            dst_pts = np.array([
                [0, 0],
                [output_w - 1, 0],
                [output_w - 1, output_h - 1],
                [0, output_h - 1]], dtype=np.float32)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if M is not None:
                # Apply the homography transformation
                warped_image = cv2.warpPerspective(frame, M, (output_w, output_h))
                # output = warped_image

    # pprint(centroids)


    cv2.imshow("standard", output)   
    if warped_image is not None: 
        cv2.imshow("warped", warped_image)
    if cv2.waitKey(1) == ord('q'):
        break
