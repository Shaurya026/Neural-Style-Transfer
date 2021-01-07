# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 04:46:31 2021

@author: shaur
"""

import tensorflow as tf

def content_cost(content_output , img_content):
    m, n_H, n_W, n_C = content_output.get_shape().as_list()
    a_C_unrolled = tf.reshape(content_output,shape = [m,-1,n_C]) # converting into (m,n_H*n_W,n_C)
    a_G_unrolled = tf.reshape(img_content,shape = [m,-1,n_C])
    J_content = tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled,a_G_unrolled))) #(1/(2*n_H*n_W*n_C))*
    return J_content

def Gram_matrix(M):
  return tf.matmul(M,M,transpose_b = True)

def style_cost(style_output, img_style):
  m, n_H, n_W, n_C = style_output.get_shape().as_list() 
  a_S = tf.transpose(tf.reshape(style_output,shape= [n_H*n_W,n_C]))
  a_G = tf.transpose(tf.reshape(img_style,shape= [n_H*n_W,n_C]))
  GS = Gram_matrix(a_S)
  GG = Gram_matrix(a_G)
  J_style_layer =  tf.reduce_sum(tf.square(tf.subtract(GS,GG))) #(1/(4*(n_C**2)*(n_H**2)*(n_W**2))) *
  return J_style_layer

def total_style_cost(style_output, img_style, style_weights):
  J_style = 0
  i = 0 
  for layer, value in style_weights :
    J_style += value * style_cost(style_output[i] , img_style[i])
    i += 1
  return J_style

def total_cost(J_content, J_style, alpha = 10, beta = 40): 
  J = alpha * J_content + beta * J_style
  return J

def loss(input_img, content_output, style_output, style_weights):
  cont_img = input_img[0]
  sty_img = input_img[1:]
  J_content = content_cost(content_output,cont_img)
  J_style = total_style_cost(style_output, sty_img, style_weights)
  J_total = total_cost(J_content, J_style,1e4,1e-2)
  return J_total


def clip_0_1(image):
  return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

@tf.function()
def training(input_image_1, model, content_output, style_output, style_weights,opt = tf.optimizers.Adam(learning_rate=0.03,beta_1=0.99, epsilon=1e-1)
):
  with tf.GradientTape() as tape:
    out = model(input_image_1)
    cost = loss(out,content_output,style_output,style_weights)
  gradient = tape.gradient(cost, input_image_1)
  opt.apply_gradients([(gradient, input_image_1)])
  input_image_1.assign(clip_0_1(input_image_1))
  