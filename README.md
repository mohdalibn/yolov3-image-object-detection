<!-- Project Name -->
# ![yolov3-image-object-detection](https://user-images.githubusercontent.com/95453430/162766553-b6a93bb1-f88d-4d31-bc6c-2856b800db68.svg)

<!-- Project Images -->
![YoloV3 Image Object Detection](https://user-images.githubusercontent.com/95453430/163043174-ce52b478-a688-4111-b1d0-87db75d52173.png)

![YoloCarTestImage](https://user-images.githubusercontent.com/95453430/163043200-d46b5613-fac6-4a03-b848-a69b1b256a5c.png)

![YoloChairsTestImage](https://user-images.githubusercontent.com/95453430/163043209-04d48c11-b588-4fa6-bae0-d5f1de518f93.png)

<!-- Project Description -->
# ![project-description (13)](https://user-images.githubusercontent.com/95453430/162766577-a350ea6a-55a9-4fef-9d28-0357c7a4ae49.svg)

This is a **Python Script** which uses YoloV3 to **detect objects in images**. YOLOv3 (You Only Look Once, Version 3) is a real-time object detection algorithm that identifies specific objects in videos, live feeds, or images. YOLO uses features learned by a deep convolutional neural network to detect an object. Versions 1-3 of YOLO were created by Joseph Redmon and Ali Farhadi <a href="https://viso.ai/deep-learning/yolov3-overview/#:~:text=YOLOv3%20(You%20Only%20Look%20Once%2C%20Version%203)%20is%20a,Joseph%20Redmon%20and%20Ali%20Farhadi.">(**Reference - viso.ai**)</a>.

<!-- Project Tech-Stack -->
# ![technologies-used (13)](https://user-images.githubusercontent.com/95453430/162766585-6546db9f-3fd1-469a-8818-64a575a0c528.svg)

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Python](https://img.shields.io/badge/yolov3-white?style=for-the-badge&logo=yolo&logoColor=#00FFFF)
![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Figma](https://img.shields.io/badge/figma-%23F24E1E.svg?style=for-the-badge&logo=figma&logoColor=white)

<!-- How To Use Project -->
# ![how-to-use-project (8)](https://user-images.githubusercontent.com/95453430/162766595-77371398-b473-4f47-844f-fa574e0ba5a1.svg)

**Install the following Python libraries in your Virtual Environment using PIP**.

*Note: The library names are **CASE-SENSITIVE** for PIP installations below. Make sure your type them correctly.*

*Install OpenCV for Python*
```Python
pip install opencv-python
```

*Install Numpy for Python*
```Python
pip install numpy
```

**Download the YoloV3-320 weights from the link below**.

https://pjreddie.com/media/files/yolov3.weights

**To check out other models, click on the link below**.

https://pjreddie.com/darknet/yolo/

Download a copy of this repository onto your local machine and extract it into a suitable folder.
- Create a Virtual Environment in that folder.
- Install all the required Python libraries mentioned above.
- **(Important) Move the downloaded YoloV3 Weights into the YoloFiles folder in the Root Directory. The script will not work without this step.**
- Open a Command Prompt/Terminal in the **Root Directory** of the Project.
- To test an image, run the following script-
```Python
python YoloImageObjectDetection.py
```
- To test a video, run the following script-
```Python
python YoloVideoObjectDetection.py
```
- There are already some images and videos provided in the **Assets** folder for testing. To test a different image/video for object detection, place the desired image/video in the **Assets** folder in the **Root Directory** and **change the file name on the 9th line of the YoloImageObjectDetection.py script in the case of an image and YoloVideoObjectDetection.py in the case of a video**.
- Enjoy playing around with the scripts!
