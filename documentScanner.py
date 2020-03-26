from pyimagesearch.transform import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import argparse
import imutils
import cv2

def showIMG(image,name):
    cv2.imshow(name,image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

ap = argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True,help="Path to the image to be scanned")
args = vars(ap.parse_args())

# reading the image
image = cv2.imread(args["image"])
orig = image.copy()

ratio = image.shape[0]/500

# edge detections to draw contours
print("[1] Detecting edges...")
image = imutils.resize(image,height=500)
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(5,5),0)
edged = cv2.Canny(blur,75,200)
showIMG(edged,"Edged")

# finding countour of largest area
print("[2] Finding contours...")
cnts = cv2.findContours(edged,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts,key=cv2.contourArea,reverse=True)[:5]

for cnt in cnts:
    peri = cv2.arcLength(cnt,True)
    approx = cv2.approxPolyDP(cnt,peri*0.02,True)

    if len(approx)==4:
        screenCnt = approx
        break

# drawing contours
cv2.drawContours(image,[screenCnt],-1,(0,255,0),2)
showIMG(image,"Outline")

# applying transformation to the image to bring it top view
print("[3] Applying perspective transformation...")
warped = four_point_transform(orig,screenCnt.reshape(4,2)*ratio)
warped = cv2.cvtColor(warped,cv2.COLOR_BGR2GRAY)
T = threshold_local(warped,11,offset=10,method="gaussian")
warped = (warped > T).astype("uint8")*255

# displaying the output image
showIMG(imutils.resize(warped,height=650),"Scanned Image")
opname="output_"+args["image"].split("/")[-1]
cv2.imwrite(opname,warped)