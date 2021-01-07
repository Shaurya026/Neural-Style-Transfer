# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 04:30:51 2021

@author: shaur
"""
import tensorflow as tf
from tensorflow.keras.applications.vgg19 import VGG19
    

def model(x,y):
    '''
    Function to return the VGG19 model with approriate output layers for GD
    '''
    model = VGG19(include_top= False,
              weights = 'imagenet',
              input_shape = (int(x),int(y),3))
    model.trainable = False
    content_block = ['block4_conv4'] 
    style_block = ['block2_conv1','block3_conv1','block4_conv1','block5_conv1']
    output_layers = content_block + style_block
    outputs = [model.get_layer(layer).output for layer in output_layers]

    model = tf.keras.Model([model.input],outputs)
    return model

def prediction(content_img):
    '''
    Parameters
    ----------
    content_img : nd array of image
        Must be an image of appropriate dimensions in RGB format
    Returns
    -------
    prediction on the image
    '''
    model = VGG19(include_top= True,
              weights = 'imagenet',)
    x = tf.keras.applications.vgg19.preprocess_input(content_img*255)
    x = tf.image.resize(x, (224, 224)) 
    probabilities = model(x)
    print(probabilities.shape)
    predictions = tf.keras.applications.vgg19.decode_predictions(probabilities.numpy())[0]
    prediction = [(class_name, prob) for (number, class_name, prob) in predictions]
    return prediction