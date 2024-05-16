#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import random
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist


# In[2]:


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


# In[4]:


def display_image(before,after,title1,title2):
    plt.subplot(1,2,1)
    plt.imshow(before, cmap='gray')
    plt.title(title1)
    plt.subplot(1,2,2)
    plt.imshow(after, cmap='gray')
    plt.title(title2)
    plt.show()


# In[7]:


def contrast_stretching(image, r1, r2, s1, s2, alpha):
    stretched_img = np.zeros(image.shape)
    m,n = image.shape
    for i in range(m):
        for j in range(n):
            intensity_values = image[i][j]
            if intensity_values <= r1:
                s = int((s1/r1)*intensity_values)
            elif r1 <= intensity_values < r2:
                s = int((s2-s1)/(r2-r1) * (intensity_values - r1) + s1)
            else:
                s = int((alpha-s2)/(255-r2) * (intensity_values - r2) + s2)
            s = max(0,min(alpha,s))
            stretched_img[i][j] = s
    return stretched_img


# In[8]:


image = x_train[43]
stretched_img = contrast_stretching(image, 80,60,200,230,255)
display_image(image, stretched_img, "before", "after")


# In[33]:


def log(image, c=1):
    m,n = image.shape
    float_img = image.astype(float)
    log_img = c * np.log1p(float_img)
    return log_img.astype(np.uint8)


# In[34]:


image = x_train[43]
log_img = log(image)
display_image(image, log_img, "before log", "after log")


# In[35]:


def powerlaw(image,c=1,gamma=0.7):
    m,n = image.shape
    float_img = image.astype(float)
    powerlaw_img = c * np.power(float_img,gamma)
    return powerlaw_img.astype(np.uint8)


# In[41]:


image = x_train[43]
powerlaw_img = powerlaw(image, c=1, gamma=0.8)
display_image(image, powerlaw_img, "before power law", "after power law")


# In[ ]:





# In[ ]:




