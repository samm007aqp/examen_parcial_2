import cv2
import math
import numpy as np 
from matplotlib import pyplot as plt

def Conservative_S(img,size):
	maxi = np.amax(img)
	rows,cols= img.shape
	out = img.copy()
	m = size//2
	for i in range(m,rows-m):
		for j in range(m,cols-m):
			value = out[i,j]
			temp = out[i-m:i+m+1,j-m:j+m+1]
			mm = temp[m,m]
			temp[m,m] = 0
			maximo = np.amax(temp)
			temp[m,m] = 255
			minimo = np.amin(temp)
			print([value,mm,maximo,minimo])
			if value> maximo:
				value = maximo
			if value< minimo:
				value = minimo
			out[i,j] = value
	return out

img= cv2.imread("man.jpg",cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (img.shape[1]//2,img.shape[0]//2)   )
out = Conservative_S(img,5)
cv2.imshow("entrada",img)
cv2.imshow("salida", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("man1.jpg",img)
cv2.imwrite("man2.jpg",out)

