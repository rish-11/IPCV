#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import random
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
from scipy.fft import fft2, ifft2, fftshift, ifftshift 


# In[3]:


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


# In[4]:


def display_image(before,after,beforetitle,aftertitle):
  plt.subplot(1,2,1)
  plt.imshow(before, cmap='gray')
  plt.title(beforetitle)
  plt.subplot(1,2,2)
  plt.imshow(after, cmap='gray')
  plt.title(aftertitle)
  plt.show()


# In[9]:


# TASK 1 = IDEAL LOW PASS FILTER

def ideal_lowpass_filter(image, threshold=30):
    row,col = image.shape
    mask = np.zeros((row,col), dtype=np.float32)
    crow,ccol = row//2, col//2
    for i in range(row):
        for j in range(col):
            D = np.sqrt((i-crow)**2 + (j-ccol)**2)
            if D <= threshold:
                mask[i][j] = 1
    
    dft = fft2(image)
    dft_shift = fftshift(dft)
    dft_shift_filtered = dft_shift * mask
    dft_shift_filtered = ifftshift(dft_shift_filtered)
    image_filtered = ifft2(dft_shift_filtered)
    image_filtered = np.abs(image_filtered)
    return image_filtered


# In[11]:


image = x_train[11]
lowpass_image = ideal_lowpass_filter(image, threshold=11)
display_image(image, lowpass_image, "Before ideal Low pass","After ideal Low pass")


# In[14]:


# TASK 2 = IDEAL HIGH PASS FILTER

def ideal_highpass_filter(image, threshold=30):
    row,col = image.shape
    mask = np.ones((row,col), dtype=np.float32)
    crow,ccol = row//2, col//2
    for i in range(row):
        for j in range(col):
            D = np.sqrt((i-crow)**2 + (j-ccol)**2)
            if D <= threshold:
                mask[i][j] = 0
    
    dft = fft2(image)
    dft_shift = fftshift(dft)
    dft_shift_filtered = dft_shift * mask
    dft_shift_filtered = ifftshift(dft_shift_filtered)
    image_filtered = ifft2(dft_shift_filtered)
    image_filtered = np.abs(image_filtered)
    return image_filtered


# In[19]:


image = x_train[11]
highpass_image = ideal_highpass_filter(image, threshold=2)
display_image(image, highpass_image, "Before ideal high pass","After ideal high pass")


# In[20]:


# TASK 3 = GAUSSIAN LOW PASS FILTER

def gaussian_lowpass_filter(image, threshold):
    row, col = image.shape
    mask = np.zeros((row,col), dtype=np.float32)
    crow, ccol = row//2, col//2
    for i in range (row):
        for j in range (col):
            D = np.sqrt((i-crow)**2 + (j-ccol)**2)
            mask[i][j] = np.exp(-(D**2)/(2*(threshold**2)))
    dft = fft2(image)
    dft_shifted = fftshift(dft)
    dft_shifted_filtered = dft_shifted * mask
    dft_shifted_filtered = ifftshift(dft_shifted)
    image_filtered = ifft2(dft_shifted_filtered)
    image_filtered = np.abs(image_filtered)
    return image_filtered


# In[25]:


image = x_train[11]
gaussian_lowpass_image = gaussian_lowpass_filter(image, threshold=30)
display_image(image, gaussian_lowpass_image, "Before gaussian low pass","After gaussian low pass")


# In[27]:


# TASK 3 = GAUSSIAN HIGH PASS FILTER

def gaussian_highpass_filter(image, threshold):
    row, col = image.shape
    mask = np.zeros((row,col), dtype=np.float32)
    crow, ccol = row//2, col//2
    for i in range (row):
        for j in range (col):
            D = np.sqrt((i-crow)**2 + (j-ccol)**2)
            mask[i][j] = 1 - np.exp(-(D**2)/(2*(threshold**2)))
    dft = fft2(image)
    dft_shifted = fftshift(dft)
    dft_shifted_filtered = dft_shifted * mask
    dft_shifted_filtered = ifftshift(dft_shifted)
    image_filtered = ifft2(dft_shifted_filtered)
    image_filtered = np.abs(image_filtered)
    return image_filtered


# In[30]:


image = x_train[11]
gaussian_highpass_image = gaussian_highpass_filter(image, threshold=30)
display_image(image, gaussian_highpass_image, "Before gaussian high pass","After gaussian high pass")


# In[ ]:




