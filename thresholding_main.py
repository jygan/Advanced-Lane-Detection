import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pickle
from perspective_regionofint_main import *
from skimage import img_as_ubyte

#grad threshold sobel x/y
# Define a function that takes an image, gradient orientation, kernel
# and threshold min / max values.
def abs_sobel_thresh(img, orient='x', sobel_kernel = 3, thresh = (0,255)):
    #hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    #l_channel = hls[:,:,1]
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize = sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize = sobel_kernel))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    # Return the result
    return binary_output

# Define a function to return the magnitude of the gradient
# for a given sobel kernel size and threshold values
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output


# Define a function to threshold an image for a given range and Sobel kernel for directionsobel
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output.astype(np.uint8)

#used for the white lanes histogram equalisation thresholding 
def adp_thresh_grayscale(gray, thr = 250):

    img = cv2.equalizeHist(gray)
    ret, thrs = cv2.threshold(img, thresh=thr, maxval=255, type=cv2.THRESH_BINARY)

    return thrs

#Color thresholding, takes saturation and value images in single channel and corresponding threshold values 
def color_thr(s_img, v_img, s_threshold = (0,255), v_threshold = (0,255)):
    s_binary = np.zeros_like(s_img).astype(np.uint8)
    s_binary[(s_img > s_threshold[0]) & (s_img <= s_threshold[1])] = 1
    v_binary = np.zeros_like(s_img).astype(np.uint8)
    v_binary[(v_img > v_threshold[0]) & (v_img <= v_threshold[1])] = 1
    col = ((s_binary == 1) | (v_binary == 1))
    return col

#the main thresholding operaion is performed here 
def thresholding(img, grad_thx_min =211, grad_thx_max =255,grad_thy_min =0, grad_thy_max = 25, mag_th_min = 150,mag_th_max = 255, dir_th_min  = 0.7, dir_th_max = 1.3, s_threshold_min = 113, s_threshold_max = 255, v_threshold_min = 234, v_threshold_max = 255,  k_size = 15, adp_thr = 250):
    # Convert to HSV color space and separate the V channel
    imshape = img.shape
    #convert to HLS
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    #read saturation channel
    s_channel = hls[:,:,2].astype(np.uint8)
    #Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float)
    #read the value channel
    v_channel = hls[:,:,2].astype(np.uint8)
    #threshold grad
    ksize = k_size # Choose a larger odd number to smooth gradient measurements
    gradx = abs_sobel_thresh(v_channel, orient='x', sobel_kernel=ksize, thresh=(grad_thx_min,grad_thx_max))
    grady = abs_sobel_thresh(v_channel, orient='y', sobel_kernel=ksize, thresh=(grad_thy_min, grad_thy_max))
    mag_binary = mag_thresh(v_channel, sobel_kernel=ksize, mag_thresh=(mag_th_min, mag_th_max))
    dir_binary = dir_threshold(v_channel, sobel_kernel=ksize, thresh=(dir_th_min, dir_th_max))
    combined = np.zeros_like(gradx)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    
    # Threshold color channel for yellow
    s_binary = color_thr(s_channel, v_channel, s_threshold=(s_threshold_min,s_threshold_max), v_threshold= (v_threshold_min,v_threshold_max)).astype(np.uint8)
    
    #histogram equalised thresholding for white lanes 
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.uint8)
    adp = adp_thresh_grayscale(gray, adp_thr)/255
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    #color_binary = np.dstack(( np.zeros_like(s_binary),combined, s_binary))
    color_binary = np.zeros_like(gradx)
    color_binary[(combined == 1) | (s_binary == 1) | (adp == 1)] = 1
    color_binary = np.dstack(( color_binary,color_binary,color_binary)).astype(np.float32)
    #region of interest is applied here 
    vertices = np.array([[(.55*imshape[1], 0.6*imshape[0]), (imshape[1],imshape[0]),
                        (0,imshape[0]),(.45*imshape[1], 0.6*imshape[0])]], dtype=np.int32)
    color_binary = region_of_interest(color_binary.astype(np.uint8), vertices)
    #plt.imshow(color_binary*255, cmap = "gray")
    return color_binary.astype(np.float32), combined, s_binary


#this is a function defined for the interactive thresholding operation hyperparameters tweeking 
#here we do not return the image but plot the image as required 
def thresholding_interative(img, grad_thx_min =211, grad_thx_max =255,grad_thy_min =0, grad_thy_max = 25, mag_th_min = 150,mag_th_max = 255, dir_th_min  = 0.7, dir_th_max = 1.3, s_threshold_min = 113, s_threshold_max = 255, v_threshold_min = 234, v_threshold_max = 255,  k_size = 15, adp_thr = 250):
    # Convert to HSV color space and separate the V channel
    imshape = img.shape

    #vertices = np.array([[(.65*imshape[1], 0.65*imshape[0]), (imshape[1],imshape[0]),
    #                    (0,imshape[0]),(.35*imshape[1], 0.65*imshape[0])]], dtype=np.int32)
    #img = region_of_interest(img, vertices)
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    s_channel = hls[:,:,2].astype(np.uint8)
    #l_channel = hls[:,:,1].astype(np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float)
    v_channel = hls[:,:,2].astype(np.uint8)
    #threshold grad
    ksize = k_size # Choose a larger odd number to smooth gradient measurements
    gradx = abs_sobel_thresh(v_channel, orient='x', sobel_kernel=ksize, thresh=(grad_thx_min,grad_thx_max))
    grady = abs_sobel_thresh(v_channel, orient='y', sobel_kernel=ksize, thresh=(grad_thy_min, grad_thy_max))
    mag_binary = mag_thresh(v_channel, sobel_kernel=ksize, mag_thresh=(mag_th_min, mag_th_max))
    dir_binary = dir_threshold(v_channel, sobel_kernel=ksize, thresh=(dir_th_min, dir_th_max))
    combined = np.zeros_like(gradx)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    
    # Threshold color channel
    s_binary = color_thr(s_channel, v_channel, s_threshold=(s_threshold_min,s_threshold_max), v_threshold= (v_threshold_min,v_threshold_max)).astype(np.uint8)
    
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.uint8)
    adp = adp_thresh_grayscale(gray, adp_thr)/255
    #plt.imshow(adp, cmap = 'gray')
    #plt.show()
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    #color_binary = np.dstack(( np.zeros_like(s_binary),combined, s_binary))
    color_binary = np.zeros_like(gradx)
    color_binary[(combined == 1) | (s_binary == 1) | (adp == 1)] = 1
    color_binary = np.dstack(( color_binary,color_binary,color_binary)).astype(np.float32)
    vertices = np.array([[(.55*imshape[1], 0.6*imshape[0]), (imshape[1],imshape[0]),
                        (0,imshape[0]),(.45*imshape[1], 0.6*imshape[0])]], dtype=np.int32)
    color_binary = region_of_interest(color_binary.astype(np.uint8), vertices)
    plt.imshow(color_binary*255, cmap = "gray")
    #return color_binary.astype(np.float32), combined, s_binary
    
