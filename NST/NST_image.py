# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 04:40:59 2021

@author: shaur
"""

# image lodaer 

import tensorflow as tf
import matplotlib.pyplot as plt
from keras.preprocessing import image
import tkinter as tk
from tkinter import filedialog
root = tk.Tk()
root.withdraw()
import numpy as np
from PIL import Image

def X_Y(X,Y):
    global x
    global y
    x = X
    y = Y
def load_img():
    """Returns loaded image in same order as asked (Both being an PIL Image object) """
    # loading of images :
    file_path_1 = filedialog.askopenfilename(title = 'Style Image')
    style_image = image.load_img(file_path_1,target_size = (int(x),int(y)))

    file_path_2 = filedialog.askopenfilename(title = 'Content Image')
    content_image = image.load_img(file_path_2,target_size = (int(x),int(y)))
    
    return style_image,content_image

def display_imges(content,style):
    """ Simple plotting"""
    # images : 
    plt.subplot(1,2,1)
    plt.axis('off')
    plt.title('content image')
    plt.imshow(content)

    plt.subplot(1,2,2)  
    plt.axis('off')
    plt.title('style image')
    plt.imshow(style)
    
    fig = plt.gcf()
    fig.set_size_inches(17, 17)
    plt.show()
    
def nd_converter(img):
    """ Returns nd array with normalized pixels"""
    return tf.keras.preprocessing.image.img_to_array(img).reshape(1,int(x),int(y),3) /255.

def initial_var():
    style,content = load_img()
    style = nd_converter(style)
    content = nd_converter(content)
    
    return tf.constant(content), tf.constant(style), tf.Variable(content) 

def tensor_to_image(tensor):
  tensor = tensor*(255)
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return Image.fromarray(tensor)


