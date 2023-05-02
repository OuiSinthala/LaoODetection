# Welcome to my object detection project
#### This project is intended to be a fun project for learning about object detection.
### REFERENCE
#### Thanks to these resources:

* (tutorial): https://www.youtube.com/watch?v=WgPbbWmnXJ8&t=2234s  
* (tracker algorithm): https://github.com/abewley/sort 


# Use case: step by step

### 1. Clone this project to your local machine.
#### Note : the location of this repository will be downloaded on the folder that you are currently on, so you should put it in the place where you can run import the package easily.
```bash
git clone https://github.com/OuiSinthala/LaoODetection.git
```

### 2. Create a new virtual environment to avoid package conflict
##### macos or linux:
```bash
python3 -m venv __your virtual environment name__
```
##### windows:
```bash
python -m venv __your virtual environment name__
```

### 3. Activate the virtual environment
###### the " " are not included
##### macos or linux:
```bash
source "your virtual environment"/bin/activate
```
##### windows:
```bash
"your virtual environment"/Scripts/activate
```

### 4. Install necessary dependencies for this package to the virtual environment.
#### Once you have created the virtual environment, you should run this command.
```bash
pip install -r requirements.txt
```
### 5. Finally: If there is no any error, that is mean that you have successfully installed the package.


## Next are some examples:

### 1. People detection
* ##### In this example, I only test with detecting only people. 
* ##### You can try other objects if there is any error:
* ##### Tell me :) and fix it together.
```python
import cv2 as cv
from LaoODetection import ObjectDetector    #import the package

def main():
    # Input source: webcam
    video_capture = cv.VideoCapture(0)

    # chose object target
    detector = ObjectDetector(target_object="person") 
    # run the object detection algorithm
    detector.object_detection(video_capture)    
    # getting data from the algorithm
    print("Detected: {0} people.".format(detector.number_of_detected_objects))

if __name__ == "__main__":
    main()
```

### 2. Counting specific object on a specific area:
* **In this example we need the region of interest in order to detect only specific location in the frame**
* **The target_object can be other objects such as person, cat, dog**

```python
import cv2 as cv
from LaoODetection import ObjectDetector

COUNTER_line1 = [80, 400, 700, 400]  # this should be changed to fit your need
region_of_interest = cv.imread("RegionOfInterest/roi_people.png")


def main():
    # Note you can change the input source to be a webcam
    # Input source: Video
    video_location = 'testing_video/cars.mp4'
    video_capture = cv.VideoCapture(video_location)

    # Running the object detection algorithm
    target = ["car", "bus", "truck", "motorbike"]
    detector = ObjectDetector(target_object=target)
    detector.object_line_counter(video_capture=video_capture, region_of_interest=region_of_interest,
                                 counter_line1=COUNTER_line1, detecting_range=(15, 30))
    # getting the result to command screen
    print("Detected: {0} people.".format(detector.number_of_detected_objects))


if __name__ == "__main__":
    main()

```

