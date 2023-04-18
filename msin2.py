import cv2
import numpy as np

# Load YOLO object detection model
net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Create ORB feature detector
orb = cv2.ORB_create(6000)

# Load video file
cap = cv2.VideoCapture("video.mp4")

# Create background subtractor
back_sub = cv2.createBackgroundSubtractorMOG2(history=200,varThreshold=70)

# Set ROI for vehicle detection
roi = np.array([[0, 200], [640, 200], [640, 480], [0, 480]])


# Initialize variables for vehicle counting
vehicle_count = 0
prev_vehicle_count = 0
colors= np.random.uniform(0, 255, size=(len(classes), 3))
while True:
    # Read frame from video file
    ret, frame = cap.read()

    # Apply background subtraction

    fg_mask = back_sub.apply(frame)
    _, fg_mask = cv2.threshold(fg_mask, 254, 255, cv2.THRESH_BINARY)

    # Apply ROI mask
    roi_mask = np.zeros_like(fg_mask)
    fg_mask_roi = cv2.bitwise_and(fg_mask, roi_mask)
    contours,_=cv2.findContours(fg_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 :
                # Object detected is a vehicle
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])
                x = int(center_x - w/2)
                y = int(center_y - h/2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Perform ORB feature detection on detected vehicle boxes
    keypoints_list = []
    for box in boxes:
        x, y, w, h = box
        roi = fg_mask_roi[y:y+h, x:x+w]
        keypoints = orb.detect(roi, None)
        keypoints_list.append(keypoints)

    # Draw detected vehicles on frame
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    for i in indices:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = confidences[i]
        color = colors[class_ids[i]]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0,0,255), 2)
        cv2.putText(frame,label+f" {i + 1} ", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cx = x + w / 2
        cy = y + h / 2
        cv2.circle(frame, (int(cx), int(cy)), 5, (0, 255, 0), -1)

        # Count vehicles that pass through ROI
        vehicle_roi_mask = np.zeros_like(fg_mask)
        vehicle_roi_mask = cv2.bitwise_and(vehicle_roi_mask, roi_mask)
        vehicle_roi_pixels = cv2.countNonZero(vehicle_roi_mask)
        if vehicle_roi_pixels >0:
            vehicle_count += 1

        # Update previous vehicle count
        prev_vehicle_count = vehicle_count


        # Display vehicle count on frame
        #cv2.putText(frame, f"Vehicle Count: {prev_vehicle_count}", (50, 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display frame with detected vehicles
        cv2.imshow("background substractio",fg_mask)
        #cv2.imshow("ROI", fg_mask_roi)
        cv2.imshow("Vehicle Detection and Counting System", frame)



        # Press 'q' to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Release video file and destroy all windows
cap.release()
cv2.destroyAllWindows()

