#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
import random


# In[2]:


(x_train, y_train), (_,_) = fashion_mnist.load_data()


# In[10]:


def display_image(before,after,title1,title2):
    plt.subplot(1,2,1)
    plt.imshow(before, cmap='gray')
    plt.title(title1)
    plt.subplot(1,2,2)
    plt.imshow(after, cmap='gray')
    plt.title(title2)
    plt.show()


# In[11]:


def gaussian_noise(image,mean=0, stddev=25):
    noise = np.random.normal(mean, stddev, image.shape)
    noisey_img = np.clip(image+noise, 0, 255).astype(np.uint8)
    return noisey_img


# In[14]:


image = x_train[43]
noise_img = gaussian_noise(image, mean=0, stddev=25)
display_image(image, noise_img, "original", "gaussian noise")


# In[18]:


from scipy.ndimage import convolve

def avg_filter(image, kernel_size=(3,3)):
    kernel = np.ones(kernel_size, dtype=np.float32) / (kernel_size[0]*kernel_size[1])
    filtered_image = convolve(image,kernel)
    return filtered_image.astype(np.uint8)
    


# In[20]:


image = noise_img
avg_filter_img = avg_filter(image, kernel_size=(3,3))
display_image(image, avg_filter_img, "before filter", "after filter")


# In[23]:


def salt_pepper_noise(image, salt_prob, pepper_prob):
    noisey_img = np.copy(image)
    salt_coords = np.random.rand(*image.shape) < salt_prob
    pepper_coords = np.random.rand(*image.shape) < pepper_prob
    noisey_img[salt_coords] = 1
    noisey_img[pepper_coords] = 0
    return noisey_img


# In[26]:


image = x_train[43]
salt_pepper_noise_img = salt_pepper_noise(image, 0.06, 0.04)
display_image(image, salt_pepper_noise_img, "Before Noise", "After Noise")


# In[27]:


def median_filter(image):
    m,n = image.shape
    filter_image = np.zeros((m,n))
    for i in range(1, m-1):
        for j in range(1, n-1):
            neighborhood = image[i-1:i+2, j-1:j+2]
            filter_image[i][j] = np.median(neighborhood)
    return filter_image.astype(np.uint8)


# In[28]:


img = salt_pepper_noise_img
filteredimg = median_filter(img)
display_image(img, filteredimg, "before filter", "after filter")


# In[ ]:




