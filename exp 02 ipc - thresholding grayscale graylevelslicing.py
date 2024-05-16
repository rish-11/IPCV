#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import random
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist


# In[2]:


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


# In[3]:


def display_image(before,after,title1,title2):
    plt.subplot(1,2,1)
    plt.imshow(before, cmap='gray')
    plt.title(title1)
    plt.subplot(1,2,2)
    plt.imshow(after, cmap='gray')
    plt.title(title2)
    plt.show()


# In[10]:


def thresholding(image, threshold=127):
    row, col = image.shape
    new_img = image.copy()
    for i in range(row):
        for j in range(col):
            if new_img[i][j] >= threshold:
                new_img[i][j] = 255
            else:
                new_img[i][j] = 0
    return new_img
            


# In[11]:


image = x_train[43]
thresholded_img = thresholding(image)
display_image(image, thresholded_img, "before", "after")


# In[21]:


def negative(img):
    temp = []
    for i in range (len(img)):
        row = []
        for j in range (len(img[i])):
            row.append(255-image[i][j])
        temp.append(row)
    return temp


# In[22]:


image = x_train[43]
negative_img = negative(image)
display_image(image, negative_img, "before", "after")


# In[27]:


def gray_level_slicing_wo_bg(image,r1,r2):
    result = np.zeros_like(image)
    for i in range(len(image)):
        for j in range (len(image[i])):
            if r1 <= image[i][j] <= r2:
                result[i][j] = image[i][j] - 1
    return result


# In[33]:


image = x_train[43]
gray_level_slicing_wo_bg_img = gray_level_slicing_wo_bg(image, 3, 8)
display_image(image, gray_level_slicing_wo_bg_img, "before", "after")


# In[34]:


def gray_level_slicing_w_bg(image,r1,r2):
    result = np.copy(image)
    for i in range(len(image)):
        for j in range (len(image[i])):
            if r1 <= image[i][j] <= r2:
                result[i][j] = image[i][j] - 1
    return result


# In[37]:


image = x_train[43]
gray_level_slicing_w_bg_img = gray_level_slicing_w_bg(image, 12, 15)
display_image(image, gray_level_slicing_w_bg_img, "before", "after")


# In[ ]:




