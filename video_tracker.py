import cv2

#video we wanna use
video_file = cv2.VideoCapture('Pedestrians Compilation.mp4')

car_tracker_file = 'cars.xml'
pedestrian_tracker_file = 'haarcascade_fullbody.xml'

car_tracker = cv2.CascadeClassifier(car_tracker_file)
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_tracker_file)

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
    pedestrians = pedestrian_tracker.detectMultiScale(grayscaled_frame)

    #same thing, draw rectangle on each car for each frame
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x+2, y+2), (x+w, y+h), (255, 0, 0), 2)    #just some extra styling
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    for (x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('Car & Pedestrian Detector', frame)
    #wait on each frame for 1ms, further returns the key
    key = cv2.waitKey(1)

    #stop video if q is pressed (cuz it goes till video ends, otherwise u must have to break it thru terminal)
    if key==81 or key==113:
        break

#release video capture, stop reading and just clean up
video_file.release()