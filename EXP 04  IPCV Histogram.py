#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import matplotlib.pyplot as plt


# In[27]:


def display_image(before,after,beforetitle,aftertitle):
  plt.subplot(1,2,1)
  plt.imshow(before, cmap='gray')
  plt.title(beforetitle)
  plt.subplot(1,2,2)
  plt.imshow(after, cmap='gray')
  plt.title(aftertitle)
  plt.show()


# In[15]:


def grayscale_image(image):
    row, col, channels = image.shape
    grayscaled_img = np.zeros((row, col), dtype=np.uint8)
    for i in range (row):
        for j in range (col):
            pixel_values = sum(image[i][j])//channels
            grayscaled_img[i][j] = pixel_values
    return grayscaled_img


# In[16]:


from keras.datasets import cifar10


# In[17]:


(x_train, y_train),(_,_) = cifar10.load_data()


# In[29]:


image = x_train[15]
plt.imshow(image)


# In[30]:


gray_img = grayscale_image(image)
plt.imshow(gray_img, cmap='gray')


# In[31]:


plt.hist(gray_img.ravel(), bins=256, range=(0,256))


# In[34]:


# TASK 1 - HISTOGRAM EQUALISATION

def histogram_equalisation(image):
    histogram, bins = np.histogram(image.ravel(), bins=256, range=(0,256))
    cdf = histogram.cumsum()
    cdf_normalized = cdf * histogram.max() / cdf.max()
    equalized_img = np.interp(image.ravel(), bins[:-1], cdf_normalized)
    equalized_img = equalized_img.reshape(image.shape)
    return equalized_img.astype(np.uint8)


# In[35]:


gray_img_hist_eq = histogram_equalisation(gray_img)
display_image(gray_img, gray_img_hist_eq, "before", "after")


# In[38]:


plt.subplot(1,2,1)
plt.hist(gray_img)
plt.subplot(1,2,2)
plt.hist(gray_img_hist_eq)


# In[40]:


# TASK 2 - HISTOGRAM DIFFERENCING

def histogram_differencing(img1,img2):
    values1 = histogram_equalisation(img1)
    values2 = histogram_equalisation(img2)
    diff = np.abs(values1-values2)
    return diff


# In[42]:


diff_hist = histogram_differencing(gray_img, gray_img_hist_eq)


# In[45]:


plt.hist(diff_hist)


# In[59]:


# TASK 3 - HISTOGRAM STRETCHED

def histogram_stretched(image, min_val=0, max_val=255):
    min_intensity = np.min(image)
    max_intensity = np.max(image)
    stretched_image = (image - min_intensity) * ((max_val - min_val)/(max_intensity - min_intensity) + min_val)
    return stretched_image.astype(np.uint8)


# In[60]:


hist_stretched_img = histogram_stretched(gray_img, min_val=0, max_val=255)


# In[61]:


plt.subplot(1,2,1)
plt.imshow(hist_stretched_img, cmap='gray')
plt.subplot(1,2,2)
plt.hist(hist_stretched_img)


# In[ ]:





# In[ ]:




