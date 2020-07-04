import numpy as np
import scipy.ndimage as ndi
from scipy.stats.mstats import mquantiles
from PIL import Image
import cv2
import random
import math

def GaussianNoise(img,mean_gaussian_noise=0.0,sigma_gaussian_noise=0.15):
	""" Applies random additive Gaussian Noise to an image.

	Args:
	img -- 32F1C image [0..1], either binary or grayscale
	mean_gaussian_noise -- mean of the random Gaussian noise (default = 0.0) 
	sigma_gaussian_noise -- sigma of the random Gaussian noise (default = 0.15)
	
	Returns:
	out -- 32F1C [0..1], grayscale image
	"""
	H,W = img.shape
	gauss = np.random.normal(mean_gaussian_noise,sigma_gaussian_noise,(H,W))
	gauss = gauss.reshape(H,W)
	gaussiannoise = np.clip(np.float32(img) + gauss,0,1)
	return gaussiannoise

def GammaCorrection(img,gamma=(.3,3.0),clip=(0.0,1.0)):
	""" Applies random Gamma correction to an image, darken or lighten it. 
	When using it with binary images, must specify clipping values.

	Args:
	img -- 32F1C image [0..1], either binary or grayscale
	gamma --  gamma value (default = (.3, 3.0)). Either float or tuple:
		* If a single float, that value will always be used as gamma.
		* If a tuple (a, b), then a random value from the range a <= x <= b will be picked per image.
	clip -- tuple indicating clipping values for input image, useful when entry image is binary (default = (0.0,1.0))
	
	Returns:
	out -- 32F1C [0..1], grayscale image
	"""
	if isinstance(gamma,float):
		gamma = 1.0/ gamma
	elif isinstance(gamma,tuple):
		gamma = 1.0 / np.random.uniform(gamma[0],gamma[1])
	else:
		raise ValueError('Parameter gamma not float nor tuple')
	gammacorrected = np.clip(img,clip[0],clip[1]) ** gamma
	return gammacorrected

def Contrast(img,c=(.5,1.5)):
	""" Modifies contrast of the image.
	Args:
	img -- 32F1C image [0..1], either binary or grayscale
	v -- value to add, (default = (-.2, .2)). Either tulpe or float:
		* If a single float, that value will always be used as constant.
		* If a tuple (a, b), then a random value from the range a <= x <= b will be picked per image.	
	
	Returns:
	out -- 32F1C [0..1] grayscale image
	"""
	if isinstance(c,tuple):
		c = np.random.uniform(c[0],c[1])
	elif not isinstance(c,float):
		raise ValueError('Parameter c not float nor tuple')
	return((img - .5) * c + .5)

def Brightness(img,c=(-.3,.3)):
	""" Modifies brightness of the image.
	Args:
	img -- 32F1C image [0..1], either binary or grayscale
	v -- value to add, (default = (-.3, .3)). Either tulpe or float:
		* If a single float, that value will always be used as constant.
		* If a tuple (a, b), then a random value from the range a <= x <= b will be picked per image.	
	
	Returns:
	out -- 32F1C [0..1] grayscale image
	"""
	if isinstance(c,tuple):
		c=np.random.uniform(c[0],c[1])
	elif not isinstance(c,float):
		raise ValueError('Parameter v not float nor tuple')
	added = np.clip(np.float32(img+c),0,1)
	return(added)

def PixelPosition(img,sigma=5.0,order=1,maxdelta=5.0):
	""" Randomly distort the pixel positions.

	Args:
	img -- 32F1C image [0..1], either binary or grayscale
	sigma -- Sigma of the gaussian filter for distorting x,y coordinates (default = 5.0)
	order -- The order of the spline interpolation, the order has to be in the range 0-5. (default = 1)	
	maxdelta -- Maximum coordinate displacement (default = 5.0)
	
	Returns:
	out -- 32F1C [0..1], either binary or grayscale
	"""
	H,W = img.shape
	deltas = np.random.rand(2,H,W)
	deltas = ndi.gaussian_filter(deltas, (0, sigma, sigma))
	deltas -= np.amin(deltas)
	deltas /= np.amax(deltas)
	deltas = (2*deltas-1) * maxdelta
	xy = np.transpose(np.array(np.meshgrid(list(range(H)), list(range(W)))), axes=[0, 2, 1])
	deltas += xy
	res_pixel_pos = ndi.map_coordinates(img, deltas, order=order, mode="reflect")
	return res_pixel_pos

def ElasticTransform(img,alpha=1750, sigma=45):
	""" Randomly warps image.

	Args:
	img -- 32F1C image [0..1], either binary or grayscale
	alpha --  (default = 1750)
	sigma -- (default = 45)	
	
	Returns:
	out -- 32F1C [0..1], either binary or grayscale
	"""
	image_shape = img.shape
	dx = np.random.uniform(-1, 1, image_shape) * alpha
	dy = np.random.uniform(-1, 1, image_shape) * alpha
	sdx = ndi.gaussian_filter(dx, sigma=sigma, mode='constant')
	sdy = ndi.gaussian_filter(dy, sigma=sigma, mode='constant')
	x, y = np.meshgrid(np.arange(image_shape[1]), np.arange(image_shape[0]))
	distorted_indices = (y + sdy).reshape(-1, 1), (x + sdx).reshape(-1, 1)
	transformed_image = ndi.map_coordinates(img, distorted_indices, mode='nearest',order=1).reshape(image_shape)
	return transformed_image

def LensBlur(img,lens_blur=(0.0,2.0)):
	""" Applies a lens blur
	Args:
	img -- 32F1C image [0..1], either binary or grayscale
	lens_blur -- controls value of blur (default = (0.0,2.0)). Either float or tuple:
		* If a single float, that value will always be used as blur factor.
		* If a tuple (a, b), then a random value from the range a <= x <= b will be picked per image.	
	
	Returns:
	out -- 32F1C [0..1] grayscale image
	"""
	if isinstance(lens_blur,tuple):
		lens_blur=np.random.uniform(lens_blur[0],lens_blur[1])
	elif not isinstance(lens_blur,float):
		raise ValueError('Parameter lens_blur not float nor tuple')
	blurred = ndi.gaussian_filter(img, lens_blur)
	return(blurred)

def LensBlurThr(img,lens_blur=(0.0,2.0)):
	""" Applies a lens blur with thresholding (otsu) and constant black pixels

	Args:
	img -- 32F1C image [0..1], either binary or grayscale
	lens_blur -- controls value of blur (default = (0.0,2.0)). Either float or tuple:
		* If a single float, that value will always be used as blur factor.
		* If a tuple (a, b), then a random value from the range a <= x <= b will be picked per image.
	
	Returns:
	out -- 32F1C [0..1] binary image
	"""
	if isinstance(lens_blur,tuple):
		lens_blur=np.random.uniform(lens_blur[0],lens_blur[1])
	elif not isinstance(lens_blur,float):
		raise ValueError('Parameter lens_blur not float nor tuple')
	blurred = ndi.gaussian_filter(img, lens_blur)
	_,bw = cv2.threshold(np.uint8(img*255),0,1,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU) 
	p = np.sum(bw) * 100.0 / np.prod(img.shape)
	t = np.percentile(blurred, p)
	lens_blur = np.float32(blurred>t)
	return lens_blur

def Sharpen(img,lens_blur=(0.0,2.0)):
	""" Applies a sharpen operation by substracting a blurred version of the image.
	Args:
	img -- 32F1C image [0..1], either binary or grayscale
	lens_blur -- controls value of blur (default = (0.0,2.0)). Either float or tuple:
		* If a single float, that value will always be used as blur factor.
		* If a tuple (a, b), then a random value from the range a <= x <= b will be picked per image.	
	
	Returns:
	out -- 32F1C [0..1] grayscale image
	"""
	blurred = LensBlur(img,lens_blur)
	res = cv2.addWeighted(img,1.5,blurred,-.5,0)
	return(res)

def AverageBlur(img,k=(2,11)):
	""" Applies an average blur
	Args:
	img -- 32F1C image [0..1], either binary or grayscale
	k --  kernel size (default = (2, 11)). Either tulpe or int:
		* If a single int, that value will always be used as kernel size.
		* If a tuple (a, b), then a random int value from the range a <= x <= b will be picked per image.	
	Returns:
	out -- 32F1C [0..1] grayscale image
	"""
	if isinstance(k,tuple):
		k=np.random.randint(k[0],k[1])
	elif not isinstance(k,int):
		raise ValueError('Parameter k not int nor tuple')
	res = cv2.blur(img,(k,k))
	return(res)

def MedianBlur(img,k=(3,11)):
	""" Applies a median blur
	Args:
	img -- 32F1C image [0..1], either binary or grayscale
	k --  kernel size (default = (3,11)). Either int or tuple:
		* If a single int, that value will always be used as kernel size. Needs to be odd.
		* If a tuple (a, b), then a random odd int value from the range a <= x <= b will be picked per image.	
	
	Returns:
	out -- 32F1C [0..1] grayscale image
	"""
	if isinstance(k,int):
		if k%2 != 1:
			raise ValueError('Parameter k not odd value')	
	elif isinstance(k,tuple):
		k=np.random.choice([x for x in range(k[0],k[1]+1) if x%2 == 1])
	else:
		raise ValueError('Parameter k not int nor tuple')
	if k!=3 and k!=5:
		img = np.uint8(255*img)
		res = cv2.medianBlur(img,k)
		res = np.float32(res) / 255.0
	else:
		res = cv2.medianBlur(img,k)
	return(res)

def MotionBlur(img, ksize=11, angle=(0,np.pi)):
	""" Applies motion blur to the image
	Args:
	img -- 32F1C image [0..1], either binary or grayscale
	ksize -- int kernel size, (default = 11)
	angle -- float angle (default = (0,pi)). Either float or tuple:
		* If a single float, that value will always be used as percentage.
		* If a tuple (a, b), then a random value from the range a <= x <= b will be picked per image.	
	
	Returns:
	out -- 32F1C [0..1] grayscale image
	"""
	if isinstance(angle,tuple):
		angle=np.random.uniform(angle[0],angle[1])
	elif not isinstance(angle,float):
		raise ValueError('Parameter angle not float nor tuple')
	kernel=np.zeros((ksize,ksize),np.float32)
	xi=int(np.round((ksize/2) + (np.cos(angle) * (ksize/2))))
	yi=int(np.round((ksize/2) + (np.sin(angle) * (ksize/2))))
	xe=int(np.round((ksize/2) + (np.cos(angle+np.pi) * (ksize/2))))
	ye=int(np.round((ksize/2) + (np.sin(angle+np.pi) * (ksize/2))))
	cv2.line(kernel, (xi, yi), (xe, ye), 1, thickness=1)
	return cv2.filter2D(img, -1, kernel / np.sum(kernel))

def Pepper(img,p=(0,.05)):
	""" Adds pepper noise to the image
	Args:
	img -- 32F1C image [0..1], either binary or grayscale
	p -- percentage of pepper noise, (default = (0, .05)). Either tulpe or float:
		* If a single float, that value will always be used as percentage.
		* If a tuple (a, b), then a random value from the range a <= x <= b will be picked per image.	
	
	Returns:
	out -- 32F1C [0..1] grayscale image
	"""
	if isinstance(p,tuple):
		p=np.random.uniform(p[0],p[1])
	elif not isinstance(p,float):
		raise ValueError('Parameter p not float nor tuple')
	c = [np.random.randint(0, i - 1, int(np.ceil(p * img.size))) for i in img.shape]
	res= img.copy()
	res[tuple(c)] = 0
	return(res)

def Salt(img,p=(0,.05)):
	""" Adds salt noise to the image
	Args:
	img -- 32F1C image [0..1], either binary or grayscale
	p -- percentage of pepper noise, (default = (0, .05)). Either tulpe or float:
		* If a single float, that value will always be used as percentage.
		* If a tuple (a, b), then a random value from the range a <= x <= b will be picked per image.	
	
	Returns:
	out -- 32F1C [0..1] grayscale image
	"""
	if isinstance(p,tuple):
		p=np.random.uniform(p[0],p[1])
	elif not isinstance(p,float):
		raise ValueError('Parameter p not float nor tuple')
	c = [np.random.randint(0, i - 1, int(np.ceil(p * img.size))) for i in img.shape]
	res= img.copy()
	res[tuple(c)] = 1
	return(res)

def JpegCompression(img, quality=(10,50)):
	""" Applies Jpeg Compression to the image
	Args:
	img -- 32F1C image [0..1], either binary or grayscale
	quality -- quality parameter of jpeg compression, (default = (10, 50)). Either tulpe or int:
		* If a single int, that value will always be used as quality.
		* If a tuple (a, b), then a random value from the range a <= x <= b will be picked per image.	
	Returns:
	out -- 32F1C [0..1] grayscale image
	"""
	if isinstance(quality,tuple):
		quality = np.random.randint(quality[0],quality[1])
	elif not isinstance(quality,int):
		raise ValueError('Parameter p not int nor tuple')
	res = np.uint8(img*255)
	_, encoded_img = cv2.imencode('.jpg', res, (cv2.IMWRITE_JPEG_QUALITY, quality))
	res = cv2.imdecode(encoded_img, cv2.IMREAD_UNCHANGED)
	res = np.float32(res/255.0)
	return res

def RandomBackground(img,scales=[1.0,5.0,10.0,20.0,50.0,100.0],weights=[1,2,4,8,16,32],alpha=.85):
	""" Blends the image with a random background
	Args:
	img -- 32F1C image [0..1], either binary or grayscale
	scales -- list of floats for the different random noise scales (default = [1.0,5.0,10.0,20.0,50.0,100.0])
	weights -- list of ints for the different weights for each scale (must have same length than scales) (default = [1.0,5.0,10.0,20.0,50.0,100.0])
	alpha -- float, blending parameter. (default = 0.85)
	
	Returns:
	out -- 32F1C [0..1] grayscale image
	"""
	h,w = img.shape
	result = ndi.zoom(np.float32(np.random.rand(int(h/scales[0]+1), int(w/scales[0]+1))),scales[0])[:h,:w]*weights[0]
	for we,s in zip(scales[1:],weights[1:]):
		result += ndi.zoom(np.random.rand(int(h/s+1), int(w/s+1)),s)[:h,:w]*we
	result -= np.min(result)
	result /= np.max(result)
	blended = alpha * img + (1-alpha) * result
	return(blended)

def Shear(img,shear=(-.5,.25),background=0):
	""" Generates a random shear of the image.
	Args:
	img -- 32F1C image [0..1], in grayscale or binary
	shear -- shear angle (default = (-0.5,0.25)). Either float or tuple:
		* If a single float, that value will always be used as shear angle.
		* If a tuple (a, b), then a random value from the range a <= x <= b will be picked per image.	
	background -- float [0..1] to use as background intensity

	Returns:
	sheared -- 32F1C [0..1], grayscale image
	"""
	H,W = img.shape
	if isinstance(shear,tuple):
		shear=np.random.uniform(shear[0],shear[1])
	elif not isinstance(shear,float):
		raise ValueError('Parameter shear not float nor tuple')
	M=np.float32([[1,shear,0],[0,1,0]])
	sheared = cv2.warpAffine(img,M,(W,H),flags=cv2.WARP_INVERSE_MAP|cv2.INTER_CUBIC,borderValue=background)
	return(sheared)

def Rotation (img,rotation=(-5.0,5.0),background=0):
	""" Generates a random rotation of the image.

	Args:
	img -- 32F1C image [0..1], in grayscale or binary
	rotation -- rotation angle (default = (-0.5,0.25)). Either float or tuple:
		* If a single float, that value will always be used as shear angle.
		* If a tuple (a, b), then a random value from the range a <= x <= b will be picked per image.
	background -- float [0..1] to use as background intensity

	Returns:
	M -- 2x3 affine transformation matrix
	rotated -- 32F1C [0..1], grayscale image
	"""
	H,W = img.shape
	if isinstance(rotation,tuple):
		rotation=np.random.uniform(rotation[0],rotation[1])
	elif not isinstance(rotation,float):
		raise ValueError('Parameter rotation not float nor tuple')
	M = cv2.getRotationMatrix2D((W/2,H/2),rotation,1)
	rotated = cv2.warpAffine(img,M,(W,H),flags=cv2.WARP_INVERSE_MAP|cv2.INTER_CUBIC,borderValue=background)
	return(rotated)


def ShearNoPad(img,shear=(-.5,.25)):
	""" Generates a random shear of the image without padding.
	Args:
	img -- 32F1C image [0..1], in grayscale or binary
	shear -- shear angle (default = (-0.5,0.25)). Either float or tuple:
		* If a single float, that value will always be used as shear angle.
		* If a tuple (a, b), then a random value from the range a <= x <= b will be picked per image.	

	Returns:
	sheared -- 32F1C [0..1], grayscale image
	"""
	if isinstance(shear,tuple):
		shear=np.random.uniform(shear[0],shear[1])
	elif not isinstance(shear,float):
		raise ValueError('Parameter shear not float nor tuple')
	pilimage = Image.fromarray(img)
	width, height = pilimage.size
	phi = math.tan(shear)
	shift_in_pixels = phi * height
	if shift_in_pixels > 0:
		shift_in_pixels = math.ceil(shift_in_pixels)
	else:
		shift_in_pixels = math.floor(shift_in_pixels)
	matrix_offset = shift_in_pixels
	if shear <= 0:
		shift_in_pixels = abs(shift_in_pixels)
		matrix_offset = 0
		phi = abs(phi) * -1
	transform_matrix = (1, phi, -matrix_offset,0, 1, 0)
	pilimage = pilimage.transform((int(round(width + shift_in_pixels)), height),Image.AFFINE,transform_matrix,Image.BICUBIC)
	pilimage = pilimage.crop((abs(shift_in_pixels), 0, width-1, height-1))
	return np.array(pilimage.resize((width, height), resample=Image.BICUBIC))

def RotationNoPad(img,rotation=(-5.0,5.0)):
	""" Generates a random rotation of the image without padding.
	Args:
	img -- 32F1C image [0..1], in grayscale or binary
	rotation -- rotation angle (default = (-0.5,0.5)). Either float or tuple:
		* If a single float, that value will always be used as shear angle.
		* If a tuple (a, b), then a random value from the range a <= x <= b will be picked per image.	

	Returns:
	rotated -- 32F1C [0..1], grayscale image
	"""
	if isinstance(rotation,tuple):
		rotation=np.random.uniform(rotation[0],rotation[1])
	elif not isinstance(rotation,float):
		raise ValueError('Parameter rotation not float nor tuple')
	rotation = -rotation
	pilimage = Image.fromarray(img)
	x = pilimage.size[0]
	y = pilimage.size[1]
	pilimage = pilimage.rotate(rotation, expand=True, resample=Image.BICUBIC)
	X = pilimage.size[0]
	Y = pilimage.size[1]
	angle_a = abs(rotation)
	angle_b = 90 - angle_a
	angle_a_rad = math.radians(angle_a)
	angle_b_rad = math.radians(angle_b)
	angle_a_sin = math.sin(angle_a_rad)
	angle_b_sin = math.sin(angle_b_rad)
	E = (math.sin(angle_a_rad)) / (math.sin(angle_b_rad)) * (Y - X * (math.sin(angle_a_rad) / math.sin(angle_b_rad)))
	E = E / 1 - (math.sin(angle_a_rad) ** 2 / math.sin(angle_b_rad) ** 2)
	B = X - E
	A = (math.sin(angle_a_rad) / math.sin(angle_b_rad)) * B
	pilimage = pilimage.crop((int(round(E+1)), int(round(A+1)), int(round(X - E -1)), int(round(Y - A -1))))
	return np.array(pilimage.resize((x, y), resample=Image.BICUBIC))

def Scale(img, scale=(.5,1.5)):
	""" Generates a random scaling of the image.

	Args:
	img -- 32F1C image [0..1], in grayscale or binary
	scale -- plus minus percentage of scaling factor (default = (0.5,1.5)). Either float or tuple:
		* If a single float, that value will always be used as shear angle.
		* If a tuple (a, b), then a random value from the range a <= x <= b will be picked per image.

	Returns:
	scaled -- 32F1C [0..1], grayscale image with different size than img
	"""
	H,W = img.shape
	if isinstance(scale,tuple):
		scale=np.random.uniform(scale[0],scale[1])
	elif not isinstance(scale,float):
		raise ValueError('Parameter scale not float nor tuple')
	scaled = cv2.resize(img,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
	return scaled

def Translation(img,delta=(-10.0,10.0),background=0):
	""" Generates a random translation of the image.

	Args:
	img -- 32F1C image [0..1], in grayscale or binary
	delta -- pixel displacements (default = (-10,10)). Either float or tuple:
		* If a single float, that value will always be used as shear angle.
		* If a tuple (a, b), then a random value from the range a <= x <= b will be picked per image.

	background -- float [0..1] to use as background intensity

	Returns:
	M -- 2x3 affine transformation matrix
	translated -- 32F1C [0..1], grayscale image
	"""
	H,W = img.shape
	if isinstance(delta,tuple):
		dx=np.random.uniform(delta[0],delta[1])
		dy=np.random.uniform(delta[0],delta[1])
	elif not isinstance(delta,float):
		raise ValueError('Parameter delta not float nor tuple')
	else:
		dx=delta
		dy=delta
	M=np.float32([[1,0,dx],[0,1,dy]])
	translated = cv2.warpAffine(img,M,(W,H),flags=cv2.WARP_INVERSE_MAP|cv2.INTER_CUBIC,borderValue=background)
	return(translated)

def PerspectiveNoPad(img,step_x=20):
	""" Generates a random perspective transform of the image without padding.
	Args:
	img -- 32F1C image [0..1], in grayscale or binary
	step_x -- 

	Returns:
	perspective -- 32F1C [0..1], grayscale image
	"""
	h,w=img.shape
	step_y = int(h * step_x / w)
	canvas = np.zeros((h+2*step_x,w+2*step_x),dtype=np.float32)
	canvas[step_x:h+step_x,step_x:w+step_x] = img
	ch,cw=canvas.shape
	A=np.array([[step_x,step_x],
		[step_x+w,step_x],
		[step_x+w,step_x+h],
		[step_x,step_x+h]],np.float32)
	B=np.array([[step_x+np.random.randint(-step_x,step_x),step_x+np.random.randint(-step_y,step_y)],
		[step_x+w+np.random.randint(-step_x,step_x),step_x+np.random.randint(-step_y,step_y)],
		[step_x+w+np.random.randint(-step_x,step_x),step_x+h+np.random.randint(-step_y,step_y)],
		[step_x+np.random.randint(-step_x,step_x),step_x+h+np.random.randint(-step_y,step_y)]],np.float32)
	M = cv2.getPerspectiveTransform(A,B)
	warped_perspective=cv2.warpPerspective(canvas,M,(cw,ch))
	img_quad = np.array([[[step_x,step_x],[w+step_x,step_x],[w+step_x,h+step_x],[step_x,h+step_x]]],dtype=np.float32)
	warped_quad =  cv2.perspectiveTransform(img_quad, M)
	xmin = int(max(warped_quad[0][0][0],warped_quad[0][3][0]))
	xmax = int(min(warped_quad[0][1][0],warped_quad[0][2][0]))
	ymin = int(max(warped_quad[0][0][1],warped_quad[0][1][1]))
	ymax = int(min(warped_quad[0][2][1],warped_quad[0][3][1]))
	img_cropped = cv2.resize(warped_perspective[ymin:ymax,xmin:xmax],(w,h))
	return(img_cropped)

def KanungoNoise(img,alpha=2.0,beta=2.0,alpha_0=1.0,beta_0=1.0,mu=.05,k=2):
	""" Applies Kanungo noise model to a binary image.
	
	T. Kanungo, R. Haralick, H. Baird, W. Stuezle, and D. Madigan. 
	A statistical, nonparametric methodology for document degradation model validation. 
	IEEE Transactions Pattern Analysis and Machine Intelligence 22(11):1209 - 1223, 2000.

	Args:
	img -- 8U1C binary image either [0..1] or [0..255]
	alpha -- controls the probability of a foreground pixel flip (default = 2.0)
	alpha_0 --  controls the probability of a foreground pixel flip (default = 1.0)	
	beta -- controls the probability of a background pixel flip (default = 2.0)
	beta_0 -- controls the probability of a background pixel flip (default = 1.0)
	mu -- constant probability of flipping for all pixels (default = 0.05)
	k -- diameter of the disk structuring element for the closing operation (default = 2)
	
	Returns:
	out -- 8U1C [0..255], binary image
	"""
	H,W = img.shape
	img = img/np.max(img)
	dist = cv2.distanceTransform(1-img, cv2.DIST_L1, 3)
	dist2 = cv2.distanceTransform(img, cv2.DIST_L1, 3)
	P = (alpha_0*np.exp(-alpha * dist**2)) + mu
	P2 = (beta_0*np.exp(-beta * dist2**2)) + mu
	distorted = img.copy()
	distorted[((P>np.random.rand(H,W)) & (img==0))] = 1
	distorted[((P2>np.random.rand(H,W)) & (img==1))] = 0
	closing = cv2.morphologyEx(distorted, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k)))
	return closing*255
