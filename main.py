import cv2 as cv
from LaoODetection import ObjectDetector

COUNTER_line1 = [80, 400, 700, 400]  # this should be changed to fit your need
region_of_interest = cv.imread("RegionOfInterest/roi_people.png")


def main():
    # Note you can change the input source to be a webcam
    # Input source: Video
    video_location = 'testing_video/people.mp4'
    video_capture = cv.VideoCapture(video_location)

    # Running the object detection algorithm
    target = ["car", "bus", "truck", "motorbike"]
    detector = ObjectDetector(target_object="person")
    detector.object_line_counter(video_capture=video_capture, region_of_interest=region_of_interest,
                                 counter_line1=COUNTER_line1, detecting_range=(15, 30))
    # detector.object_detection(video_capture)

    print("Detected: {0} people.".format(detector.number_of_detected_objects))


if __name__ == "__main__":
    main()
