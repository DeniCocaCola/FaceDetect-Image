#
# @DeniCocaCola
# Python 3.7.2
#

import cv2

imgLocation = input("Image location: ")
img = cv2.imread(imgLocation,1)

cascade_Location = input("Cascade Classifier path: ")
face_cascade = cv2.CascadeClassifier(cascade_Location)

minNeighbors = 5
minNeighbors = input("Min Neighbors Value(Default = 5): ") #Change if it's detecting false positives


userImgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faceCoord = face_cascade.detectMultiScale(userImgGray, scaleFactor = 1.05, minNeighbors = int(minNeighbors))

for x,y,w,h in faceCoord:
    img = cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0),3) #(0, 255, 0) is RGB for the color of the outline

cv2.imshow(imgLocation, img)
print("Image Dimensions: ")
print(img.shape)
print("Detected face locations: ")
print(faceCoord)

cv2.waitKey(0)
cv2.destroyAllWindows()



