import cv2

#video we wanna use
video_file = cv2.VideoCapture('Tesla Autopilot Dashcam Compilation 2018 Version.mp4')

classifier_file = 'cars.xml'

car_tracker = cv2.CascadeClassifier(classifier_file)

#what a video is, is basically a bunch of pictures, we loop forever till video stops
while True:
    #read current frame of videos, loops and reads thru each frame returns - whether read was successful or not, and the frame
    #tuple unpacking
    (read_succssful, frame) = video_file.read()

    if read_succssful:
        #if we got our frame, we do the same, convert it to black n white
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    #detect all of our grayscaled frame
    cars = car_tracker.detectMultiScale(grayscaled_frame)

    #same thing, draw rectangle on each car for each frame
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    cv2.imshow('Car Detector', frame)
    #wait on each frame for 1ms
    cv2.waitKey(1)