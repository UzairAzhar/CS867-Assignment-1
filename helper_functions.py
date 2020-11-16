# -*- coding: utf-8 -*-
"""
@author: UzairAzhar
"""

# -*- coding: utf-8 -*-
"""
@author: UzairAzhar
"""
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import skimage
from skimage.util import random_noise
import scipy.ndimage
from mpl_toolkits.mplot3d import Axes3D 
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

disp_size=10

def load_images_from_folder(path):
    images = []
    for img_number,img_name in enumerate(os.listdir(path)):
        image=[]
        image=cv2.imread(os.path.join(path,img_name))
        images.append(image)
        
    return images

# In[] : 2.1        
def plot_image(images):
    plt.figure(figsize=(20, 20))
    no=len(images)
    no=100+no*10+1
    for i in range(len(images)):
        plt.subplot(no)
        plt.imshow(images[i])
        no+=1
        #plt.show()
  
# In[] : 2.2
def rgbExclusion(image):
    
    b,g,r=cv2.split(image)
    plt.figure(figsize=(disp_size, disp_size))
    
    ax=plt.subplot(1,3,1)
    ax.set_title("Blue Image")
    plt.imshow(b)
    
    ax=plt.subplot(1,3,2)
    ax.set_title("Green Image")
    plt.imshow(g)
    
    ax=plt.subplot(1,3,3)
    ax.set_title("Red Image")
    plt.imshow(r)
    
    
    return b,g,r
    
# In[] : 2.3
def hist_equalization(image):# take all images in directory
    
    #Converting to GrayScale
    for i in range(3):
        img1=image[i]
        plt.figure(figsize=(disp_size, disp_size))
        ax=plt.subplot(1,4,1)
        ax.set_title("GrasScale Image")
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        plt.imshow(img1)
        
        ax=plt.subplot(1,4,2)
        ax.set_title("Histogram Image")
        hist,bins = np.histogram(img1.flatten(),256,[0,256])
        plt.hist(img1.flatten(),256,[0,256])
        plt.xlim([0,256])
        
        ax=plt.subplot(1,4,3)
        ax.set_title("After Equalization Image")
        img1_eq=cv2.equalizeHist(img1)
        plt.imshow(img1_eq)
        
        ax=plt.subplot(1,4,4)
        ax.set_title("Equalized Image Histogram")
        hist,bins = np.histogram(img1_eq.flatten(),256,[0,256])
        plt.hist(img1_eq.flatten(),256,[0,256])
        plt.xlim([0,256])
        
    

def own_convolution(image,kernel):  
       

    # Add zero padding to the input image
      image_padded = np.zeros((image.shape[0] + 2, image.shape[1] + 2))
      image_padded[1:-1, 1:-1] = image

     # convolution output
      output = np.zeros_like(image) 
    # Loop over every pixel of the image
      for x in range(image.shape[1]):
        for y in range(image.shape[0]):
            # element-wise multiplication of the kernel and the image
            output[y, x] = (kernel * image_padded[y: y+3, x: x+3]).sum()*1/9
            
      return output
# In[] : 2.4 #convolution operation from scratch
def convolution(image):
      image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      plt.figure(figsize=(disp_size, disp_size))
      ax=plt.subplot(1,3,1)
      ax.set_title("Orignal GrayScale Image ")
      plt.imshow(image)
      
      #Sharpeness Conv
      sharpening_kernel=np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
      output=own_convolution(image,sharpening_kernel)
      scratch_output_convo_sharpening_kernel=output
      ax=plt.subplot(1,3,2)
      ax.set_title("Sharpening Effect Image ")
      plt.imshow(scratch_output_convo_sharpening_kernel)
      
      #Blurness Conv
      blurring_kernel=np.array([[1,1,1],[1,1,1],[1,1,1]]) # Box blur
      output=own_convolution(image,blurring_kernel)
      scratch_output_convo_blurring_kernel=output
      ax=plt.subplot(1,3,3)
      ax.set_title("Blurry Effect Image ")
      plt.imshow(scratch_output_convo_blurring_kernel)
      
      #Comparing Own Conv with Libs Conv
      import scipy.signal
      out=scipy.signal.convolve2d(image, sharpening_kernel, mode='same')
      out_libs_sharpening_kernel=np.uint8(out/9)
      
      #diff btw libs and from scratch convolution
      diff=(out_libs_sharpening_kernel-scratch_output_convo_sharpening_kernel).sum()
      return diff
  

def box_filter_conv(images):
    plt.figure(figsize=(disp_size, disp_size))
    orig=3*2 # 3 images, 2 types of images ; orig and blurred
    no=100+orig*10+1
        
    for i in range(3):
        img=images[i]
        ax=plt.subplot(no)
        ax.set_title("Orignal Image ")
        plt.imshow(img)
        no+=1
        
        dst = cv2.boxFilter(img,-1,(5,5))
        ax=plt.subplot(no)
        ax.set_title("BoxFiltered Image ")
        plt.imshow(dst)
        no+=1

def gaussian_filter_conv(images):
    plt.figure(figsize=(disp_size, disp_size))
   
    img=images[1]
    ax=plt.subplot(241)
    ax.set_title("Orignal Image ")
    plt.imshow(img)
    
    dst = cv2.GaussianBlur(img, (5, 5), 0)
    ax=plt.subplot(242)
    ax.set_title("GaussBlurr-1 Image ")
    plt.imshow(dst)
    
    dst = cv2.GaussianBlur(img, (15, 15),0)
    ax=plt.subplot(243)
    ax.set_title("GaussBlurr-2 Image ")
    plt.imshow(dst)
    
    dst = cv2.GaussianBlur(img, (35, 35),0)
    ax=plt.subplot(244)
    ax.set_title("GaussBlurr-3 Image ")
    plt.imshow(dst)
    
    img=images[0]
    ax=plt.subplot(245)
    ax.set_title("Orignal Image ")
    plt.imshow(img)
    
    dst = cv2.GaussianBlur(img, (5, 5), 0)
    ax=plt.subplot(246)
    ax.set_title("GaussBlurr-1 Image ")
    plt.imshow(dst)
    
    dst = cv2.GaussianBlur(img, (15, 15),0)
    ax=plt.subplot(247)
    ax.set_title("GaussBlurr-2 Image ")
    plt.imshow(dst)
    
    dst = cv2.GaussianBlur(img, (35, 35),0)
    ax=plt.subplot(248)
    ax.set_title("GaussBlurr-3 Image ")
    plt.imshow(dst)
    

def add_noise(images):
    plt.figure(figsize=(disp_size, disp_size))
    sp_noise_images=[]
    gauss_noise_images=[]
    for i in range(len(images)):
        image=images[i]
        
# =============================================================================
#         # Adding Salt and Pepper Noises
# =============================================================================
        saltpepper_noise = random_noise(image, mode='s&p',amount=0.1)
        saltpepper_noise = np.array(255*saltpepper_noise, dtype = 'uint8')
        sp_noise_images.append(saltpepper_noise)
        
# =============================================================================
#         # Adding Gaussian Noises
# =============================================================================
        gauss_noise=random_noise(image, mode='gaussian', seed=None,var=.05)
        gauss_noise = np.array(255*gauss_noise, dtype = 'uint8')
        gauss_noise_images.append(gauss_noise)
    
# =============================================================================
#         #Plotting
# =============================================================================
        ax=plt.subplot(131)
        ax.set_title("Orignal Image ")
        plt.imshow(images[0])
        
        ax=plt.subplot(132)
        ax.set_title("Gaussian Noise Image ")
        plt.imshow(gauss_noise_images[0])
        
        ax=plt.subplot(133)
        ax.set_title("Salt and Pepper Noise Image ")
        plt.imshow(sp_noise_images[0])
        
    return  sp_noise_images,gauss_noise_images
    
def gaussianFilter_and_medianFilters(images,sp_noise_images,gauss_noise_images):
    plt.figure(figsize=(disp_size, disp_size))
    Gauss_filt_size=9
    Med_filt_size=9
    GaussianFilt_sp=[]
    GaussianFilt_gauss=[]
    MedianFilt_sp=[]
    MedianFilt_gauss=[]
    image_num=0
    
    for i in range(len(images)):
        sp_noise_image=sp_noise_images[i]
        gauss_noise_image=gauss_noise_images[i]
        
# =============================================================================
#         #Gaussian Filter
# =============================================================================
        GaussianFilt_sp.append(cv2.GaussianBlur(sp_noise_image, (Gauss_filt_size, Gauss_filt_size), 0))
        GaussianFilt_gauss.append(cv2.GaussianBlur(gauss_noise_image, (Gauss_filt_size, Gauss_filt_size), 0))
        
# =============================================================================
#         Median Filter
# =============================================================================
        MedianFilt_sp.append(cv2.medianBlur(sp_noise_image,Med_filt_size))
        MedianFilt_gauss.append(cv2.medianBlur(gauss_noise_image,Med_filt_size))
        
# =============================================================================
#         #Plotting
# =============================================================================
    ax=plt.subplot(241)
    ax.set_title("Orignal Image ")
    plt.imshow(images[image_num])
    
    ax=plt.subplot(242)
    ax.set_title("Gaussian Noise Image ")
    plt.imshow(gauss_noise_images[image_num])
    
    ax=plt.subplot(243)
    ax.set_title("Gaussian Filter on GaussNoise Image ")
    plt.imshow(GaussianFilt_gauss[image_num])
    
    ax=plt.subplot(244)
    ax.set_title("Median Filter on GaussNoise Image ")
    plt.imshow(MedianFilt_gauss[image_num])
    
    ax=plt.subplot(245)
    ax.set_title("Orignal Image ")
    plt.imshow(images[image_num])
    
    ax=plt.subplot(246)
    ax.set_title("Salt and Pepper Noise Image ")
    plt.imshow(sp_noise_images[image_num])
    
    ax=plt.subplot(247)
    ax.set_title("Gaussian Filter on S&P Image ")
    plt.imshow(GaussianFilt_sp[image_num])
    
    ax=plt.subplot(248)
    ax.set_title("Median Filter on S&P Image ")
    plt.imshow(MedianFilt_sp[image_num])
    
def plot(data,ax,y_len,x_len,title,fig):
    
    Y = np.arange(0, y_len, 1)
    X = np.arange(0, x_len, 1)
    X, Y = np.meshgrid(X, Y)
    el=0
    az=-40
    
    
    ax.view_init(elev=el, azim=az)
    ax.set_title(title)
    surf = ax.plot_surface(X, Y, data, cmap=cm.gist_rainbow,
                           linewidth=0, antialiased=False)
    ax.set_zlabel('Z-Axis')
    ax.set_xlabel('X-Axis')
    ax.set_ylabel('Y-Axis')
    #ax.set_zlim(0,256)
    
def mesh_plots(images):
    for i in range(len(images)):
        img=images[i]
        gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        y_len=np.shape(gray)[0]
        x_len=np.shape(gray)[1]
        #Gaussian Filter with different window size
        img1_g_1=cv2.GaussianBlur(gray, (5, 5),0)
        img1_g_2=cv2.GaussianBlur(gray, (15, 15),0)
        img1_g_3=cv2.GaussianBlur(gray, (29, 29),0)
        
        fig1 = plt.figure(figsize=(disp_size, disp_size))
        ax1 = fig1.add_subplot(131)
        ax1.imshow(img)
    # =============================================================================
    #     #meshPlots of Gaussian Filter with varying sigma
    # =============================================================================
        fig = plt.figure(figsize=(disp_size, disp_size))
        ax = fig.add_subplot(331, projection='3d')
        plot(img1_g_1,ax,y_len,x_len,'Gauss Filter Window 5x5',fig)
        ax = fig.add_subplot(332, projection='3d')
        plot(img1_g_2,ax,y_len,x_len,'Gauss Filter Window 15x15',fig)
        ax = fig.add_subplot(333, projection='3d')
        plot(img1_g_3,ax,y_len,x_len,'Gauss Filter Window 29x29',fig)
        
    # =============================================================================
    #     #meshPlots of First Order Derivatibe of Gaussian Filter with varying sigma
    # =============================================================================
        #First Order Derivative of Gaussian --> Aprrox by Sobel
         #Gaussian Filter with different window size
        vertical_filter=np.array([[-1 ,0,1],[-1 ,0,1],[-1 ,0,1]])*1/3
        horizontal_filter=np.array([[1 ,1,1],[0,0,0],[-1 ,-1,-1]])*1/3
        img1_g_1_x=cv2.filter2D(img1_g_1,-1,vertical_filter)
        img1_g_1_y=cv2.filter2D(img1_g_1,-1,horizontal_filter)
        img2_s_1=np.uint8(np.sqrt(img1_g_1_x**2+img1_g_1_y**2))
        
        
        img1_g_2_x=cv2.filter2D(img1_g_2,-1,vertical_filter)
        img1_g_2_y=cv2.filter2D(img1_g_2,-1,horizontal_filter)
        img2_s_2=np.uint8(np.sqrt(img1_g_2_x**2+img1_g_2_y**2))
        
        img1_g_3_x=cv2.filter2D(img1_g_3,-1,vertical_filter)
        img1_g_3_y=cv2.filter2D(img1_g_3,-1,horizontal_filter)
        img2_s_3=np.uint8(np.sqrt(img1_g_3_x**2+img1_g_3_y**2))
    
        
        ax = fig.add_subplot(334, projection='3d')
        plot(img2_s_1,ax,y_len,x_len,'Deriv of Gauss Filter Window 1x1',fig)
        ax = fig.add_subplot(335, projection='3d')
        plot(img2_s_2,ax,y_len,x_len,'Deriv of Gauss Filter Window 3x3',fig)
        ax = fig.add_subplot(336, projection='3d')
        plot(img2_s_3,ax,y_len,x_len,'Deriv of Gauss Filter Window 5x5',fig)
    

# =============================================================================
#     #meshPlots of Gaussian Filter with varying sigma
# =============================================================================
        img3_l_1=cv2.Laplacian(img2_s_1,-1)
        img3_l_2=cv2.Laplacian(img2_s_2,-1)
        img3_l_3=cv2.Laplacian(img2_s_3,-1)
        
        ax = fig.add_subplot(337, projection='3d')
        plot(img3_l_1,ax,y_len,x_len,'Laplacian Gauss Filter Window 5x5',fig)
        ax = fig.add_subplot(338, projection='3d')
        plot(img3_l_2,ax,y_len,x_len,'Laplacian of Gauss Filter Window 15x15',fig)
        ax = fig.add_subplot(339, projection='3d')
        plot(img3_l_3,ax,y_len,x_len,'Laplacian of Gauss Filter Window 29x29',fig)
        
        ax1 = fig1.add_subplot(132)
        ax1.set_title("First derivative of Gaussian Filtered Img")
        ax1.imshow(img2_s_1)
        ax1 = fig1.add_subplot(133)
        ax1.set_title("Laplacian of Gaussian Filtered Img")
        ax1.imshow(img3_l_1)
        
def sobel(images):
    for i in range(len(images)):
        img=images[i]
        gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        filter_v=np.array([[1,   0,  -1],[  2,   0,  -2],[      1,   0,  -1]])
        filter_h=np.array([[1,   2,  1],[  0,0,0],[      -1,   -2,  -1]])
        
        x=cv2.filter2D(img,-1,filter_v)
        x=np.uint8(np.absolute(x))
        y=cv2.filter2D(img,-1,filter_h)
        y=np.uint8(np.absolute(y))
        mag=np.uint8(np.absolute(cv2.Sobel(img,cv2.CV_64F,1,1)))
        
        plt.figure(figsize=(disp_size, disp_size))
        ax=plt.subplot(1,4,1)
        ax.set_title("Orignal Image ")
        plt.imshow(img)
        
        ax=plt.subplot(1,4,2)
        ax.set_title("X-derivative Img")
        plt.imshow(x)
        
        ax=plt.subplot(1,4,3)
        ax.set_title("y-derivative Img")
        plt.imshow(y)
        
        ax=plt.subplot(1,4,4)
        ax.set_title("Magnitude Img")
        plt.imshow(mag)

        
def laplacian(images):
    for i in range(len(images)):
        img=images[i]
        img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
        img1_g_1=cv2.GaussianBlur(img, (5, 5),0)
        img1_g_2=cv2.GaussianBlur(img, (15, 15),0)
        img1_g_3=cv2.GaussianBlur(img, (29, 29),0)
        
        laplace_1=np.uint8(np.absolute(cv2.Laplacian(img,cv2.CV_8U)))
        laplace_2=np.uint8(np.absolute(cv2.Laplacian(img,cv2.CV_8U)))
        laplace_3=np.uint8(np.absolute(cv2.Laplacian(img,cv2.CV_8U)))
        
        log_1=np.uint8(np.absolute(cv2.Laplacian(img1_g_1,cv2.CV_8U)))
        log_2=np.uint8(np.absolute(cv2.Laplacian(img1_g_2,cv2.CV_8U)))
        log_3=np.uint8(np.absolute(cv2.Laplacian(img1_g_3,cv2.CV_8U)))
        
        
        plt.figure(figsize=(disp_size, disp_size))
        ax=plt.subplot(3,3,1)
        ax.set_title("Orignal Image ")
        plt.imshow(img)
        
        ax=plt.subplot(3,3,2)
        ax.set_title("Filtered Image with 5x5 kernel")
        plt.imshow(log_1)
        
        ax=plt.subplot(3,3,3)
        ax.set_title("Laplace Magnitude with 5x5 kernel")
        plt.imshow(laplace_1)
        
        ax=plt.subplot(3,3,4)
        ax.set_title("Orignal Image ")
        plt.imshow(img)
        
        ax=plt.subplot(3,3,5)
        ax.set_title("Filtered Image with 15x15 kernel")
        plt.imshow(log_2)
        
        ax=plt.subplot(3,3,6)
        ax.set_title("Laplace Magnitude with 15x15 kernel")
        plt.imshow(laplace_2)
        
        ax=plt.subplot(3,3,7)
        ax.set_title("Orignal Image ")
        plt.imshow(img)
        
        ax=plt.subplot(3,3,8)
        ax.set_title("Filtered Image with 29x29 kernel")
        plt.imshow(log_3)
        
        ax=plt.subplot(3,3,9)
        ax.set_title("Laplace Magnitude with 29x29 kernel")
        plt.imshow(laplace_3)
        
def canny(images):
    
    for i in range(len(images)):
        img=images[i]

        canny=cv2.Canny(img,120,150)
        
        plt.figure(figsize=(disp_size, disp_size))
        ax=plt.subplot(1,2,1)
        ax.set_title("Orignal Image ")
        plt.imshow(img)
        
        ax=plt.subplot(1,2,2)
        ax.set_title("Canny Edge Detector Response")
        plt.imshow(canny)
