#!/usr/bin/env python
# coding: utf-8

# In[74]:


import numpy as np
import random
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist


# In[75]:


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


# In[76]:


def display_image(before,after,beforetitle,aftertitle):
  plt.subplot(1,2,1)
  plt.imshow(before, cmap='gray')
  plt.title(beforetitle)
  plt.subplot(1,2,2)
  plt.imshow(after, cmap='gray')
  plt.title(aftertitle)
  plt.show()


# In[79]:


image.shape


# In[80]:


# TASK 1 - REGION GROWING

def region_growing(image, seed, threshold):
    row, col = image.shape
    x, y = seed
    region = np.zeros_like(image, dtype=bool)
    region[x,y] = True
    region_intensity = image[x,y]
    to_process = [(x,y)]
    while to_process:
        x, y = to_process.pop(0)
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            xn, yn = x+dx, y+dy
            if 0 <= xn < row and 0 <= yn < col and not region[xn,yn]:
                if abs(image[xn,yn] - region_intensity) <= threshold:
                    region[xn,yn] = True
                    to_process.append((xn, yn))
    return region


# In[81]:


image = x_train[11]
seed = (4, 4)
threshold = 0.1
region_growing_image = region_growing(image, seed, threshold)
display_image(image, region_growing_image, "Before Region Growing", "After Region Growing")


# In[86]:


from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()


# In[93]:


import numpy as np

def region_growing(image, seed, threshold):
    height, width= image.shape[:2]  # Adjust for 3D shape of CIFAR-10 images
    x, y = seed
    region = np.zeros((height, width), dtype=bool)  # Only consider 2D shape
    region[x, y] = True
    region_intensity = image[x, y]  # Calculate mean intensity for RGB
    to_process = [(x, y)]
    while to_process:
        x, y = to_process.pop(0)
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            xn, yn = x + dx, y + dy
            if 0 <= xn < height and 0 <= yn < width and not region[xn, yn]:
                pixel_intensity = image[xn, yn]  # Calculate mean intensity for RGB
                if np.linalg.norm(region_intensity - pixel_intensity) <= threshold:
                    region[xn, yn] = True
                    to_process.append((xn, yn))
    return region

image = x_train[11]
seed = (4, 4)
threshold = 0.1
region_growing_image = region_growing(image, seed, threshold)
display_image(image, region_growing_image, "Before Region Growing", "After Region Growing")


# In[ ]:




