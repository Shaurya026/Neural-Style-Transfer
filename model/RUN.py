import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os 

from keras.preprocessing import image
import tkinter as tk
from tkinter import filedialog
root = tk.Tk()
root.withdraw()

# ******************************************************************************************************
# loading of images :
file_path_1 = filedialog.askopenfilename()
style_image = image.load_img(file_path_1,target_size = (224,224))

file_path_2 = filedialog.askopenfilename()
content_image = image.load_img(file_path_2,target_size = (224,224))

# images : 
plt.subplot(1,2,1)
plt.axis('off')
plt.title('content image')
plt.imshow(content_image)

plt.subplot(1,2,2)
plt.title('style image')
plt.imshow(style_image)

# *******************************************************************************************************
'''conversion of image into tensors followed by loading of model'''
# converting image to numpy n-dimensional vector
content_img = tf.keras.preprocessing.image.img_to_array(content_image).reshape(1,224,224,3) /255.
style_img = tf.keras.preprocessing.image.img_to_array(style_image).reshape(1,224,224,3) / 255.

from tensorflow.keras.applications.vgg19 import VGG19
model = VGG19(include_top= True,
              weights = 'imagenet')
# content_image = load_img('/content/cat.jpg')
# style_image = load_img('/content/sandstone.jpg')

# input : here for content image
x = tf.keras.applications.vgg19.preprocess_input(content_img*255)
x = tf.image.resize(x, (224, 224))
model.trainable = False

# ******************************************************************************************************************************#
'''declaring of layers to be used for encoding of data and calculation of cost'''

# taking block4_conv4 as the intermdeiate layer for decoding content_image:
# content_block = ['block4_conv4'] # this doesn't give that good representation of content so we will use : 
content_block = ['block4_conv4'] # if layer is near deep end image won't be good 
                                 # and if near start than style won't be good

# taking 4 intermediate layers for decoding style_image :
style_block = ['block1_conv2','block2_conv2','block3_conv2','block4_conv2','block5_conv2']
# added block1_conv1 in start to get euclidian loss for colors in the style

output_layers = content_block + style_block
outputs = [model.get_layer(layer).output for layer in output_layers]

model = tf.keras.Model([model.input],outputs)
#model.summary()

# ********************************************************************************************************************************#
'''Finally making the outputs of respective layers we defined above for both the style image and content image'''

# intializing the final image : 
'''intializing with content_image makes it easier to upgrade at each step '''
input_image_1 = tf.Variable(content_img) # it has to be a variable to make it's value changeable

# calculating encodings for content and style to use in cost function : 
content_output = model(content_img) # total 5 different tensor outputs
content_output = content_output[0]

style_output = model(style_img) # again 5 outputs
style_output = style_output[1:]

# *******************************************************************************************************************************

'''converts and displays the image matrix array '''
from PIL import Image
# for getting the image back : 
def tensor_to_image(tensor):
  tensor = tensor*(255)
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return Image.fromarray(tensor)

d = tensor_to_image(input_image_1.numpy())
#img = Image.fromarray(d, 'RGB')
plt.imshow(d)

#*******************************************************************************************************************************

'''Finds the total content cost of the image w.r.t. our provided content image '''
# making content cost function :
def content_cost(content_output , img_content):
    m, n_H, n_W, n_C = content_output.get_shape().as_list()
    # Reshape both for same size comaprison : 
    a_C_unrolled = tf.reshape(content_output,shape = [m,-1,n_C])
    a_G_unrolled = tf.reshape(img_content,shape = [m,-1,n_C])
    # computing cost :
    J_content = (1/(4*n_H*n_W*n_C))*tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled,a_G_unrolled)))
    return J_content

#***********************************************************************************************************************************

''' Calculation of Gram Mtrix : '''
# Making gram matrix for style cost : 
def Gram_matrix(M):
  return tf.matmul(M,M,transpose_b = True)

'''Style Cost of each respective layer'''
# determining Style Cost : 
def style_cost(style_output, img_style):
  m, n_H, n_W, n_C = style_output.get_shape().as_list() 
  # Again reshaping for suitable cost detemination(n_C, n_H*n_W)
  a_S = tf.transpose(tf.reshape(style_output,shape= [n_H*n_W,n_C]))
  a_G = tf.transpose(tf.reshape(img_style,shape= [n_H*n_W,n_C]))
  # Gram Matrices :
  GS = Gram_matrix(a_S)
  GG = Gram_matrix(a_G)
  # Computing the cost
  J_style_layer = (1/(4*(n_C**2)*(n_H**2)*(n_W**2))) * tf.reduce_sum(tf.square(tf.subtract(GS,GG)))
  return J_style_layer

  '''Style cost of all the layers'''
  # these weights are used for calculating total cost of style function :
  # estimating our hyperparameter of weights : 
style_weights = [
    (style_block[0], 0.2),     # you can set 
    (style_block[1], 0.2),     # any values for the 
    (style_block[2], 0.2),     # hyperparameter of 
    (style_block[3], 0.2),     # weights 
    (style_block[4], 0.2)]

# computing total style cost : 
def total_style_cost(style_output, img_style, style_weights):
  J_style = 0
  i = 0 
  for layer, value in style_weights :
    J_style += value * style_cost(style_output[i] , img_style[i])
    i += 1
  return J_style

#***********************************************************************************************************************
''' Total cost produced '''
# finally making our total cost : 
def total_cost(J_content, J_style, alpha = 10, beta = 40): 
  # alpha Beta are hyperparameters for total cost
  J = alpha * J_content + beta * J_style
  return J

'''Providing a loss computing funtion for Gradient tape in later use '''
# defining a function which encloses all the above function into one : 
def loss(input_img, content_output, style_output, style_weights):
  cont_img = input_img[0]
  sty_img = input_img[1:]
  J_content = content_cost(content_output,cont_img)
  J_style = total_style_cost(style_output, sty_img, style_weights)
  J_total = total_cost(J_content, J_style,10,200) # we can apply alpha beta values here too ! 

  return J_total

#*************************************************************************************************************************
'''defining training step for one iteration'''

# first defining out optimizer : 
opt = tf.optimizers.Adam(learning_rate=0.017)#, beta_1=0.99, epsilon=1e-1)

# to simplify our image output for each iteration
def clip_0_1(image):
  return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

# now making the function to make gradient descent on the image we want to turn into artistics style
# each function call will make a gradient descent step on the image 
@tf.function()
def training(input_image_1, model, content_output, style_output, style_weights):
  with tf.GradientTape() as tape:
    out = model(input_image_1)
    cost = loss(out,content_output,style_output,style_weights)
  #print(cost)
  # applying gradient descent :
  gradient = tape.gradient(cost, input_image_1)
  opt.apply_gradients([(gradient, input_image_1)])
  input_image_1.assign(clip_0_1(input_image_1))

#***************************************************************************************************************************

# Running the whole modeule : 
import time
start = time.time()
epochs = int(input('number of epochs :'))
steps_per_epoch = int(input('steps per epoch :'))

os.mkdir(os.path.join(os.getcwd(),r'save images'))
step = 0
for n in range(epochs):
  for m in range(steps_per_epoch):
    step += 1
    training(input_image_1, model, content_output, style_output, style_weights)
    print(".", end='')
    file_name = 'stylized-image{}.png'.format(n*1000 + m)
    file_save = os.path.join(os.path.join(os.getcwd(),r'save images'),file_name)
    tensor_to_image(input_image_1).save(file_save)

  print("Train step: {}".format(step))
end = time.time()
print("Total time: {:.1f} seconds".format(end-start))


data = tensor_to_image(input_image_1.numpy())
#img = Image.fromarray(data, 'RGB')
plt.imshow(data)