import cv2
import numpy as np 
import gausian




def getPoints(filename): 
	image = cv2.imread(filename)
	rows, cols, channel= image.shape
	origin = image.copy()
	image = cv2.resize(image, (image.shape[1]//2,image.shape[0]//2)   )
	gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	#cv2.imshow("gray", gray)
	blurred = gausian.Convolucion(gray,gausian.gaussian_blur(5,1) )
	#cv2.imshow("blurred", blurred)
	edged = cv2.Canny(blurred,100,150)
	kernel = np.ones((5, 5))
	dilated = gausian.dilation(edged, kernel)
	dilated = gausian.dilation(dilated, kernel)
	img_ths = gausian.erosion(dilated, kernel)
	#dilated = cv2.dilate(edged, kernel, iterations=2)
	#img_ths = cv2.erode(dilated, kernel, iterations=1)
	#cv2.imshow("edged",edged)
	#cv2.imshow("closing",img_ths)

	contours,hierarchy = cv2.findContours(img_ths, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	contours = sorted(contours,key= cv2.contourArea,reverse=True)
	# Area del papel con respecto al documento total
	MAX_COUNTOUR_AREA = (img_ths.shape[1]- 10) * (img_ths.shape[0] - 10 )
	maxAreaFound = MAX_COUNTOUR_AREA * 0.25
	pageContour = np.array([[[5,5]], [[5,img_ths.shape[0]-5]],[[img_ths.shape[1]-5, img_ths.shape[0]-5]],[[img_ths.shape[1]-5,5]]])
	for c in contours:
		p = cv2.arcLength(c,True)
		approx = cv2.approxPolyDP(c,0.03*p,True)

		if len(approx) == 4 and  cv2.isContourConvex(approx) :
			if  maxAreaFound < cv2.contourArea(approx) and cv2.contourArea(approx) < MAX_COUNTOUR_AREA :
				maxAreaFound = cv2.contourArea(approx)
				pageContour = approx
				break


	return approx*2



print(getPoints("recibo.jpg"))
