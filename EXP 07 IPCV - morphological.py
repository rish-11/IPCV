#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import matplotlib.pyplot as plt
import random
from keras.datasets import fashion_mnist
from keras.datasets import cifar10


# In[10]:


(x_train, y_train), (_,_) = fashion_mnist.load_data()


# In[15]:


def grayscale(image):
    row, col, channels = image.shape
    grayscaled_img = np.zeros((row, col))
    for i in range (row):
        for j in range (col):
            pixel_values = sum(image[i][j])//channels
            grayscaled_img[i][j] = pixel_values
    return grayscaled_img


# In[16]:


def display_image(before,after,beforetitle,aftertitle):
    plt.subplot(1,2,1)
    plt.imshow(before, cmap='gray')
    plt.title(beforetitle)
    plt.subplot(1,2,2)
    plt.imshow(after, cmap='gray')
    plt.title(aftertitle)
    plt.show()


# In[21]:


# TASK 1 - EROSION

def erosion(image, kernel):
    eroded_image = np.zeros_like(image)
    padded_image = np.pad(image, [(1,1), (1,1)], mode='constant', constant_values=0)
    for i in range (1, padded_image.shape[0] - 1):
        for j in range (1, padded_image.shape[1] - 1):
            if np.all(padded_image[i-1:i+2 , j-1:j+2])[kernel==1]:
                eroded_image[i-1, j-1] = 1
    return eroded_image


# In[22]:


image = x_train[43]
kernel = 1
eroded_image = erosion(image, kernel)
display_image(image, eroded_image, "Before Erosion", "After Erosion")


# In[23]:


(x_train, y_train), (_,_) = cifar10.load_data()


# In[36]:


og_img = x_train[14]
plt.imshow(og_img)


# In[37]:


og_gray_img = grayscale(og_img)
plt.imshow(og_gray_img, cmap='gray')


# In[42]:


og_gray_eroded_img = erosion(og_gray_img, kernel)
display_image(og_gray_img, og_gray_eroded_img, "Before Erosion", "After Erosion")


# In[78]:


(x_train, y_train), (_,_) = fashion_mnist.load_data()


# In[79]:


def dilation(image, kernel):
    dilated_image = np.zeros_like(image)
    padded_image = np.pad(image, [(1,1),(1,1)], mode='constant', constant_values=0)
    for i in range (1, padded_image.shape[0]-1):
        for j in range(1, padded_image.shape[1]-1):
            if np.any(padded_image[i-1:i+2 , j-1:j+2])[kernel==1]:
                dilated_image[i-1,j-1] = 1
    return dilated_image


# In[80]:


image = x_train[43]
kernel = 1
dilated_image = dilation(image, kernel)
display_image(image, dilated_image, "Before Dilation", "After Dilation")


# In[72]:


(x_train, y_train), (_,_) = cifar10.load_data()


# In[73]:


og_img = x_train[43]
plt.imshow(og_img)


# In[74]:


og_gray = grayscale(og_img)
plt.imshow(og_gray, cmap='gray')


# In[75]:


dilated_image = dilation(og_gray, kernel=1)
display_image(og_gray, dilated_image, "Before Dilation", "After Dilation")


# In[81]:


(x_train, y_train), (_,_) = fashion_mnist.load_data()


# In[82]:


def opening(image):
    eroded_image = erosion(image, kernel=1)
    dilated_image = dilation(eroded_image, kernel=1)
    return dilated_image


# In[83]:


image = x_train[43]
opened_image = opening(image)
display_image(image, opened_image, "Before Opening", "After Opening")


# In[84]:


def closing(image):
    dilated_image = dilation(image, kernel=1)
    eroded_image = erosion(dilated_image, kernel=1)
    return eroded_image


# In[85]:


image = x_train[43]
closed_image = closing(image)
display_image(image, closed_image, "Before Closing", "After Closing")


# In[88]:


def hit_miss_transform(image, foreground_kernel, background_kernel):
    hit_miss_img = np.zeros_like(image)
    padded_image = np.pad(image, [(1,1), (1,1)], mode='constant', constant_values=0)
    for i in range(1, padded_image.shape[0]-1):
        for j in range(1, padded_image.shape[1]-1):
            if np.all(padded_image[i-1:i+2 , j-1:j+2][foreground_kernel==1]) and \
            np.all(1 - padded_image[i-1:i+2, j-1:j+2][background_kernel==1]):
                hit_miss_img[i-1, j-1] = 1
    return hit_miss_img


# In[93]:


image = x_train[11]
foreground_kernel = np.array([[0,1,0],
                             [1,1,1],
                             [0,1,0]])
background_kernel = np.array([[1,0,1],
                             [0,0,0],
                             [1,0,1]])
hit_miss_img = hit_miss_transform(image,foreground_kernel,background_kernel)
display_image(image, hit_miss_img, "Before Hit Miss", "After Hit Miss")


# In[ ]:




