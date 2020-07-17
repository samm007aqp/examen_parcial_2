import cv2
import matplotlib.pyplot as plt
import numpy as np
#programa coordenada de bordes funciona y investigar como funciona tanto filtro
#https://bretahajek.com/2017/01/scanning-documents-photos-opencv/



imagen = cv2.imread("papel.jpg")
rows, cols, channel = imagen.shape
print([rows,cols])
#img = np.resize( img, (img.shape[0]//10, img.shape[1]//10) )

def resizes(imagen,rows,cols):
	return  cv2.resize(imagen,(cols//2, rows//2))

# convert to grayscale
if rows > 800 and cols > 700:
	color_img = cv2.resize(imagen, (imagen.shape[1]//2,imagen.shape[0]//2))
else :
	color_img = imagen.copy()
#color_img = resize(img)
img = cv2.cvtColor(color_img, cv2.COLOR_RGB2GRAY)

rows, cols = img.shape
print([rows,cols])
# bilateral filter preserv edges 
img2 = cv2.bilateralFilter(img,4,25,40,borderType = cv2.BORDER_CONSTANT)
cv2.imshow("bilateral filter",img2)
#Create black and white image based on adaptive threshold
#img2 = cv2.equalizeHist(img)
##cv2.imshow("equalizeHist",img2)
img2 = cv2.adaptiveThreshold(img2, 255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 61,7)
cv2.imshow("adaptiveThe",img2)
#Median filter clears small details
#img2 = cv2.medianBlur(img2,11)
##cv2.imshow("medianBlur",img2)
#add black border in case that page is touching an image border
img2 = cv2.copyMakeBorder(img2,5,5,5,5,cv2.BORDER_CONSTANT, value= [0,0,0])
##cv2.imshow("copyMkaeBoder",img2)
edges = cv2.Canny(img2,150,200)

cv2.waitKey(0)
cv2.destroyAllWindows()

#getting contours 
contours, hierarchy = cv2.findContours(edges , cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

height = edges.shape[0]
width = edges.shape[1]
MAX_COUNTOUR_AREA = (width- 10) * (height - 10 )

maxAreaFound = MAX_COUNTOUR_AREA * 0.25
pageContour = np.array([[[5,5]], [[5,height-5]],[[width-5, height-5]],[[width-5,5]]])
print([MAX_COUNTOUR_AREA, maxAreaFound])

for cnt in contours:
	perimeter = cv2.arcLength(cnt,True)
	approx = cv2.approxPolyDP(cnt, 0.03 * perimeter, True)

	if len(approx) == 4 and  cv2.isContourConvex(approx):
		#print(cv2.contourArea(approx)) inprimir el area de cuadrados generados por findContou..
		if  maxAreaFound < cv2.contourArea(approx) and cv2.contourArea(approx) < MAX_COUNTOUR_AREA :
			maxAreaFound = cv2.contourArea(approx)
			pageContour = approx
print("------------------------------------------------")
print(pageContour)
color = (255, 0, 0)
for pnt in pageContour:
	print(pnt)
	cv2.circle(color_img,(pnt[0,0],pnt[0,1]),10,color)

plt.subplot(121),plt.imshow(cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR) ),plt.title("input") 
plt.subplot(122),plt.imshow(cv2.cvtColor(edges, cv2.COLOR_RGB2BGR)),plt.title("bilateral")
plt.show()
#print(pageContour)

print("llllllllllllllllllllllllllllllllllll")
def fourCornersSort(pts):
	diff = np.diff(pts, axis=1)
	summ = pts.sum(axis=1)	
	return np.array([pts[np.argmin(summ)],pts[np.argmax(diff)],pts[np.argmax(summ)],pts[np.argmin(diff)]])

def contourOffset(cnt,offset):
	cnt += offset
	cnt[cnt<0] = 0
	return cnt 


pageContour = fourCornersSort(pageContour[:,0])
#print(pageContour)
pageContour = contourOffset(pageContour,(-5,5))
#print(pageContour)
sPoints = pageContour
#sPoints = pageContour.dot(rows/ 800) esta linea va con el resize del principio

height = max(np.linalg.norm(sPoints[0] - sPoints[1]),np.linalg.norm(sPoints[2] - sPoints[3]))	
width = max(np.linalg.norm(sPoints[1] - sPoints[2]),np.linalg.norm(sPoints[3] - sPoints[0]))

#create target points
tPoints = np.array([[0, 0],	[0, height],[width, height],[width, 0]], np.float32)
sPoints = np.float32(sPoints)
#Wraping perspective
#print(sPoints)
#print(tPoints)
#sPoints = 4*sPoints
#tPoints = 4*tPoints
#print(sPoints)
#print(tPoints)
for p in sPoints:
	print(p)
	cv2.circle(imagen,(int(p[0]*2),int(p[1]*2) ),10,color)
cv2.imshow("imagen_real", imagen)
cv2.waitKey(0)

M= cv2.getPerspectiveTransform(sPoints,tPoints)
newImage = cv2.warpPerspective(imagen,M,(int(width),int(height)))

#cv2.imshow("salida_affinetrasform",newImage)
#cv2.waitKey(0)

#print(pageContour)
#cv2.imwrite("edges.jpg", edges)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

