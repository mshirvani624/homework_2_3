import cv2
import numpy as np

# Programmer Mohammad Shirvani
# Date 1402/02/06
# This code detect face and blurs it

detector=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

cap=cv2.VideoCapture(0)

while True:
	ret, frame= cap.read()
	
	if ret:

		frame=cv2.flip(frame,1)
		gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		results=detector.detectMultiScale(gray)
		for (x,y,w,h) in results:
			cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)

			H, W = gray.shape

			kernel_size = 25

			kernel = np.ones((kernel_size,kernel_size))

			blur=gray.copy()

			for i in range(y,y+h):
				for j in range(x,x+w):
					img_littel=gray[i:i+kernel_size,j:j+kernel_size]

					littel_float=np.float32(img_littel)

					blur[i,j]=(littel_float*kernel).mean()

			blur = blur.astype(np.uint8)
		
		cv2.imshow("webcamasdfasdf",blur)
		q=cv2.waitKey(1)
		if q==ord('q'):
			break
			
cap.release()			
cv2.destroyAllWindows()	