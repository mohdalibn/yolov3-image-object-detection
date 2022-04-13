
# THIS SCRIPT IS USED TO TEST YOLOV3 ON VIDEOS

# Libraries required to run this project
import cv2
import numpy as np

# Loading the Video that we want to test
cap = cv2.VideoCapture('Assets/teslacam.mp4')
cap.set(3, 640)
cap.set(4, 480)


WidthTarget = 320  # the yolo weights are of the images of size 320x320

classesFile = 'YoloFiles/coconames.txt'
classNames = []

with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
    # print(classNames)
    # print(len(classNames))


# loading in the YOLO config file and Weights
modelConfiguration = 'YoloFiles/YOLOv3-320Config.cfg'
modelWeights = 'YoloFiles/YOLOV3-320Weights.weights'

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)

# Using opencv as the backend and that we want to use the cpu
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


def findObjects(outputs, frame):
    Height, Width, _ = frame.shape
    bounding_box = []
    ClassIds = []
    confidence_values = []
    confidenceThreshold = 0.5
    nmsThreshold = 0.3

    for output in outputs:
        for detection in output:
            # we neglect the first 5 values in the list cuz those are the center x, center y, width, height, and confidence that an object is present
            scores = detection[5:]
            # this returns the index of the max probability
            classId = np.argmax(scores)
            confidence = scores[classId]

            if confidence > confidenceThreshold:
                # The width and height of the box is at index 2, and 3 of the list. The values are in decimal form and so we have to multiply it with our image heighter and width
                width, height = int(
                    detection[2]*Width), int(detection[3]*Height)

                center_x = int(detection[0]*Width - width/2)
                center_y = int(detection[1]*Height - height/2)

                bounding_box.append([center_x, center_y, width, height])
                ClassIds.append(classId)
                confidence_values.append(float(confidence))

    # among the overlapping bounding boxes for a similar object, it's going to pick the box with the highest confidence score and remove the rest. This returns the indices of the box that was picked
    indices = cv2.dnn.NMSBoxes(
        bounding_box, confidence_values, confidenceThreshold, nmsThreshold)

    for i in indices:
        i = i[0]
        box = bounding_box[i]
        x, y, w, h = box[0], box[1], box[2], box[3]

        # draws the rectangular box around the object
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)

        # displays the predicted class of the object
        cv2.putText(frame, f'{classNames[ClassIds[i]].upper()} {int(confidence_values[i]*100)}%',
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)


# Creating a while loop to get the frames of the webcam
run = True
while run:
    sucess, frame = cap.read()

    # We cannot input the plain image that we recieve from a video or our webcam into our network. It only accepts the blob format

    # converting image to blob
    blob = cv2.dnn.blobFromImage(
        frame, 1/255, (WidthTarget, WidthTarget), [0, 0, 0], 1, crop=False)

    net.setInput(blob)

    # getting the names of the output layers
    layerNames = net.getLayerNames()
    # print(layerNames)
    net.getUnconnectedOutLayers()  # this line gets the output layers. This returns the tensor/multi-dim-array of indices which we can use to plug in layerNames and get the output layer names. This doesn't follow the standard of starting from index 0, so we should subtract 1 from any index to get the names layerNames

    outputNames = [layerNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    # print(outputNames)

    # now, we can send this image as a forward pass to our network and we can find the output of these 3 layers

    outputs = net.forward(outputNames)
    # print(len(outputs)) # prints 3 cuz we are getting 3 different outputs

    # print(type(outputs)) # our output is basically a list, so we can use the list operation like list[0] access elements

    # print(outputs[0].shape) # prints (300, 85). The first output layer produces 300 bounding boxes. The last 80 among the 85 are the probability predictions of different classes. the remaning 5 are the bounding boxes's center x, center y, width, height, and confidence that there is an object present

    # print(outputs[1].shape) # prints (1200, 85) the second output layer produces 1200 bounding boxes

    # print(outputs[2].shape) # prints (4800, 85) the third output layer produces 4800 bounding boxes

    findObjects(outputs, frame)

    new_frame = cv2.resize(frame, (720, 480))
    cv2.imshow('YoloV3 Output Image', new_frame)

    key = cv2.waitKey(1)
    if key == 81 or key == 113:
        cap.release()
        cv2.destroyAllWindows()
        break
