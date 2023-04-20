# Welcome to my object detection project


### Create a new virtual environment for preventing conflict
macos:
```bash
python3 -m venv __your virtual environment name__
```
windows:
```bash
python -m venv __your virtual environment name__
```

### Install this package
```bash
pip install LaoODtectoin
```

### Install necessary dependencies
```bash
pip install -r requirements.txt

```

### Example for detecting people on the screen

```python
import cv2 as cv
from src.LaoODetection import ObjectDetector

region_of_interest = cv.imread("RegionOfInterest/roi_people.png")


def main():
    # Note you can change the input source to be a webcam
    # Input source: Video 
    video_location = 'testing_video/people.mp4'
    video_capture = cv.VideoCapture(video_location)

    # Running the object detection algorithm
    detector = ObjectDetector(model="yoloV8m.pt", target_object="person")
    detector.object_detection(video_capture)

    print("Detected: {0} people.".format(detector.number_of_detected_objects))


if __name__ == "__main__":
    main()
```

### Example for counting specific object:
* **In this example we need the region of interest in order to detect only specific location in the frame**
* **The target_object can be other objects such as person, cat, dog**

```python
import cv2 as cv
from src.LaoODetection import ObjectDetector

COUNTER_line1 = [80, 400, 700, 400]  # this should be changed to fit your need
region_of_interest = cv.imread("RegionOfInterest/roi_car.png")


def main():
    # Note you can change the input source to be a webcam
    # Input source: Video
    video_location = 'testing_video/cars.mp4'
    video_capture = cv.VideoCapture(video_location)

    # Running the object detection algorithm
    target = ["car", "bus", "truck", "motorbike"]
    detector = ObjectDetector(model="yoloV8m.pt", target_object=target)
    detector.object_line_counter(video_capture=video_capture,
                                 region_of_interest=region_of_interest,
                                 counter_line1=COUNTER_line1,
                                 detecting_range=(15, 30))
    print("Detected amount: {0}.".format(detector.number_of_detected_objects))


if __name__ == "__main__":
    main()

```

### REFERENCE
Thanks to these source of information and the python module:

* https://www.youtube.com/watch?v=WgPbbWmnXJ8&t=2234s  
* https://github.com/abewley/sort 