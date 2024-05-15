import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import easyocr
img=cv2.imread("D:/Programmation/Projects Computer Vision Enginner/Plate recognition using OCR and Python/plate.png")
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img2=cv2.cvtColor(gray,cv2.COLOR_RGB2BGR)
#cv2.imshow("i",gray)

bfliter=cv2.bilateralFilter(gray,11,17,17)
edged=cv2.Canny(bfliter,30,200)
#cv2.imshow("e",edged)
keypoints=cv2.findContours(edged,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

contours=keypoints[0]
#print(len(contours))
contours=sorted(contours,key=cv2.contourArea,reverse=True)[:10]
#print(len(contours))
#cv2.drawContours(img,contours,-1,(0,87,25),2)
#cv2.imshow("Image",img)
location=None
for contour in contours:
    aprox=cv2.approxPolyDP(contour,10,True)
    if len(aprox)==4:
        location=aprox
        break

mask=np.zeros(gray.shape,np.uint8)
new_image=cv2.drawContours(mask,[location],0,255,-1)
new_image=cv2.bitwise_and(img,img,mask=mask)
#cv2.imshow('i', new_image)
(x,y)=np.where(mask==255)
#print((x,y))
(x1,y1)=(np.min(x),np.min(y))
(x2,y2)=(np.max(x),np.max(y))
cropped_image=gray[x1:x2+1,y1:y2+1]
reader=easyocr.Reader(['en'])
result=reader.readtext(cropped_image)
result
#cv2.imshow('i', cropped_image)

text=result[0][-2]
font=cv2.FONT_HERSHEY_SIMPLEX
res=cv2.putText(img,text=text,org=(aprox[0][0][0],aprox[1][0][1]+20),fontFace=font, fontScale=1,color=(0,255,0),thickness=2,lineType=cv2.LINE_AA)
res=cv2.rectangle(img,tuple(aprox[0][0]),tuple(aprox[2][0]),(0,255,0),3)

#plt.imshow(cv2.cvtColor(res,cv2.COLOR_BGR2RGB))

cv2.imshow('i', res)



cv2.waitKey(0)
cv2.destroyAllWindows()