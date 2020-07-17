import numpy as np
import cv2
from matplotlib import pyplot as plt 


def GetPerspectiveTransform(pi,ps):
	x1,y1 = pi[0]
	x2,y2 = pi[1]
	x3,y3 = pi[2]
	x4,y4 = pi[3]
	xx1,yy1 = ps[0]
	xx2,yy2 = ps[1]
	xx3,yy3 = ps[2]
	xx4,yy4 = ps[3]
	A = np.array([  [x1,y1,1,0,0,0,  -x1*xx1, -xx1*y1 ],
					[0,0,0,x1,y1,1,  -x1*yy1, -y1*yy1],
					[x2,y2, 1,0,0,0, -x2*xx2, -xx2*y2], 
					[0,0,0,x2,y2,1,  -x2*yy2, -y2*yy2],
					[x3,y3,1,0,0,0,  -x3*xx3, -xx3*y3],
					[0,0,0,x3,y3,1,  -x3*yy3, -y3*yy3],
					[x4,y4,1,0,0,0,  -x4*xx4, -xx4*y4],
					[0,0,0,x4,y4,1,  -x4*yy4, -y4*yy4]
					 ],np.float32)
	#print(A)
	ps = ps.flatten()
	x = cv2.solve(A,ps)
	x = x[1].flatten()
	x = np.append(x,np.float32(1))
	M = x.reshape(3,3)
	return M

def mapp(h):
    h = h.reshape((4,2))
    hnew = np.zeros((4,2),dtype = np.float32)

    add = h.sum(1)
    hnew[0] = h[np.argmin(add)]
    hnew[2] = h[np.argmax(add)]

    diff = np.diff(h,axis = 1)
    hnew[1] = h[np.argmin(diff)]
    hnew[3] = h[np.argmax(diff)]

    return hnew

def Eucli_Dist(x1,y1,x2,y2):
	return np.sqrt((x1-x2)**2 + (y1-y2)**2)

def Gausian_Curve(x,sigma):
	x1 = (2*np.pi)* (sigma**2)
	x2 = np.exp(-(x**2)/(2*sigma**2))
	return (1.0/x1) * x2
def Gaussian_filter(d,sigma):
	x2 = np.exp(-(d**2)/(2*sigma**2))
	return x2

def bilateral_filter(source, filtered_image, s_x, s_y, size, sigma_i, sigma_g):
	rows , cols = source.shape
	offset = size//2
	Js = 0 #resulting pixel intensity
	Ks = 0
	for i in range(size):
		for j in range(size):
			px = s_x - (offset - i)
			py = s_y - (offset - j)
			if px < rows and py < cols:
				fps = Gaussian_filter(Eucli_Dist(px, py, s_x, s_y), sigma_g)
				gps = Gaussian_filter(source[px][py] - source[s_x][s_y] ,sigma_i)
				w = fps * gps
				Js += source[px][py] * w
				Ks += w
	Js = Js / Ks
	filtered_image[s_x][s_y] = int(round(Js))

def my_bilateral_filter(source, kernel_size, sigma_i, sigma_g):
    filtered_image = np.zeros(source.shape)
    rows, cols = source.shape
    for i in range(rows): 
        for j in range(cols):# source[0]
            bilateral_filter(source, filtered_image, i, j, kernel_size, sigma_i, sigma_g)
    return filtered_image

def gaussian_blur(size,sigma):
	kernel = np.zeros((size,size),np.float32)
	m=size//2
	for x in range(-m,m+1):
		for y in range(-m,m+1):
			#x1 = sigma*(2*np.pi)**2
			x1 = (2*np.pi)*(sigma**2)
			x2 = np.exp(-(x**2+y**2)/(2*sigma**2) )
			kernel[x+m,y+m] = (1/x1) * x2
	return kernel


def Convolucion(img,kernel):
	maxi = np.amax(img)
	rows,cols= img.shape
	m,n = kernel.shape
	new= np.zeros( (rows+m-1,cols+n-1) ) 
	n=n//2
	m=m//2
	filtered_img = np.zeros(img.shape)
	new[m:new.shape[0]-m,n:new.shape[1]-n] = img
	for i in range(m,new.shape[0]-m):
		for j in range(n,new.shape[1]-n):
			temp = new[i-m:i+m+1,j-m:j+m+1]
			result = temp*kernel
			filtered_img[i-m,j-n] = result.sum()
	maximo = np.amax(filtered_img)
	c = maxi/maximo  
	filtered_img = filtered_img*c

	return filtered_img.astype(np.uint8)

def erosion(original,kernel):
	copia =  original.copy()
	row,col = original.shape
	r,c = kernel.shape
	total = np.sum(kernel)*255 - 200
	height = r//2
	width = c//2
	for y in range (height,row-height):
		for x in range(width,col-width):
			result = mul(original[y-height:y+height+1,x-width:x+width+1],kernel)
			if result > total:
				copia[y,x] = 255 
			else :
				copia[y,x] = 0
	return copia


def dilation(original,kernel):
	copia =  original.copy()
	row,col = original.shape
	r,c = kernel.shape
	total = np.sum(kernel)*255 - 10
	height = r//2
	width = c//2
	for y in range (height,row-height):
		for x in range(width,col-width):
			result = mul(original[y-height:y+height+1,x-width:x+width+1],kernel)
			if result > 250:
				copia[y,x] = 255 
			else :
				copia[y,x] = 0	
	return copia

def mul(m1,m2):
	x = np.multiply(m1,m2)
	return np.sum(x)

img= cv2.imread("recibo.jpg",cv2.IMREAD_GRAYSCALE)
#img = img.astype(int)
#k = gaussian_blur(5,2)
#x = cv2.getGaussianKernel(3,1)
#mean_filter=np.ones((3,3))/9
#output = Convolucion(img,k)
print(len(img))
print(len(img[0]) )
output = my_bilateral_filter(img,5,50,16)
output = output.astype(np.uint8)
#print(Gaussian_filter(200,8))
#print(Gaussian_filter(2,8))
#img = img.astype(np.uint8)
cv2.imwrite("salidaBiFil.jpg",output)
cv2.imshow("entrada",img)
cv2.imshow("salida", output)
cv2.waitKey(0)
cv2.destroyAllWindows()




#plt.imshow(img,cmap='gray')
#plt.figure()
#plt.imshow(output,cmap='gray')
#plt.show()