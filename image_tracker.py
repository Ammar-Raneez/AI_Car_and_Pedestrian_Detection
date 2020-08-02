import cv2

#image we gonna use
img_file = 'car.jpg'

#our pre=trained car classifier
classifier_file = 'cars.xml'

#create opencv image
img = cv2.imread(img_file)

#create car classifier
#running a cascade of haar features that we wanna run it through
#classifier is basically classifying something as a face, car etc.. in this case the car
#creates a classifier with our file
car_tracker = cv2.CascadeClassifier(classifier_file)

#convert to grayscale(cuz its faster)
black_n_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#detect cars, by passing our black n white image into our car classifier
#detect multiple scale means to detect cars of any size
#it picks out all the positions of the image where the haar features returns true
cars = car_tracker.detectMultiScale(black_n_white)
#cars holds the coordinates of the squares which are around the cars, so, using this we can draw the squares
#[[top left coordinate, width of square, height of square]]
#[[x, y, w, h]]

#draw rectangles around the cars, to make the detection visual
for (x, y, w, h) in cars:   #we unpack each array in car, so instead of doing for arr in car and arr[0] bla bla we can unpack it
    #(where to draw, (startX, startY), (currentX+width(endX), currentY+height(endY)), color(bgr), thickness of rectangle)
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)


#display the image with cars spotted
#window name, what to show
cv2.imshow('Car Detector', img)

#dont autoclose, only close upon a key press, cuz usually its only a split second
cv2.waitKey()

print("Code completed")
