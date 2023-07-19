import tensorflow as tf
import keras
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.initializers import glorot_uniform

class FeatureExtractor(tf.keras.Model):

    def __init__(self, l2=0):
        super().__init__()

        initial_weights = tf.keras.initializers.glorot_uniform(42)
        regularizer = tf.keras.regularizers.l2(l2)
        input_shape = (None, None, 3)
    
        # First two convolutional blocks are frozen (not trainable)
        self.block1_conv1=Conv2D(name = "block1_conv1", input_shape = input_shape, kernel_size = (3,3), strides = 1, filters = 64, padding = "same", activation = "relu", kernel_initializer = initial_weights, bias_initializer=initial_weights,  trainable = False)
        self.block1_conv2=Conv2D(name = "block1_conv2", kernel_size = (3,3), strides = 1, filters = 64, padding = "same", activation = "relu", kernel_initializer = initial_weights, bias_initializer=initial_weights, trainable = False)
        self.block1_pool=MaxPooling2D(name='block1_pool', pool_size = 2, strides = 2)

        self.block2_conv1=Conv2D(name = "block2_conv1", kernel_size = (3,3), strides = 1, filters = 128, padding = "same", activation = "relu", kernel_initializer = initial_weights,bias_initializer=initial_weights, trainable = False)
        self.block2_conv2=Conv2D(name = "block2_conv2", kernel_size = (3,3), strides = 1, filters = 128, padding = "same", activation = "relu", kernel_initializer = initial_weights, bias_initializer=initial_weights,trainable = False)
        self.block2_pool=MaxPooling2D(name='block2_pool', pool_size = 2, strides = 2)

        # Weight decay begins from these layers onward: https://github.com/rbgirshick/py-faster-rcnn/blob/master/models/pascal_voc/VGG16/faster_rcnn_end2end/train.prototxt
        self.block3_conv1=Conv2D(name = "block3_conv1", kernel_size = (3,3), strides = 1, filters = 256, padding = "same", activation = "relu", kernel_initializer = initial_weights,bias_initializer=initial_weights, kernel_regularizer = regularizer)
        self.block3_conv2=Conv2D(name = "block3_conv2", kernel_size = (3,3), strides = 1, filters = 256, padding = "same", activation = "relu", kernel_initializer = initial_weights,bias_initializer=initial_weights, kernel_regularizer = regularizer)
        self.block3_conv3=Conv2D(name = "block3_conv3", kernel_size = (3,3), strides = 1, filters = 256, padding = "same", activation = "relu", kernel_initializer = initial_weights,bias_initializer=initial_weights, kernel_regularizer = regularizer)
        self.block3_pool=MaxPooling2D(name='block3_pool', pool_size = 2, strides = 2)

        self.block4_conv1=Conv2D(name = "block4_conv1", kernel_size = (3,3), strides = 1, filters = 512, padding = "same", activation = "relu", kernel_initializer = initial_weights,bias_initializer=initial_weights, kernel_regularizer = regularizer)
        self.block4_conv2=Conv2D(name = "block4_conv2", kernel_size = (3,3), strides = 1, filters = 512, padding = "same", activation = "relu", kernel_initializer = initial_weights,bias_initializer=initial_weights, kernel_regularizer = regularizer)
        self.block4_conv3=Conv2D(name = "block4_conv3", kernel_size = (3,3), strides = 1, filters = 512, padding = "same", activation = "relu", kernel_initializer = initial_weights,bias_initializer=initial_weights, kernel_regularizer = regularizer)
        self.block4_pool=MaxPooling2D(name='block4_pool', pool_size = 2, strides = 2)

        self.block5_conv1=Conv2D(name = "block5_conv1", kernel_size = (3,3), strides = 1, filters = 512, padding = "same", activation = "relu", kernel_initializer = initial_weights,bias_initializer=initial_weights, kernel_regularizer = regularizer)
        self.block5_conv2=Conv2D(name = "block5_conv2", kernel_size = (3,3), strides = 1, filters = 512, padding = "same", activation = "relu", kernel_initializer = initial_weights, bias_initializer=initial_weights,kernel_regularizer = regularizer)
        self.block5_conv3=Conv2D(name = "block5_conv3", kernel_size = (3,3), strides = 1, filters = 512, padding = "same", activation = "relu", kernel_initializer = initial_weights,bias_initializer=initial_weights, kernel_regularizer = regularizer)

    def call(self, input_image):
        y = self.block1_conv1(input_image)
        y = self.block1_conv2(y)
        y = self.block1_pool(y)

        y = self.block2_conv1(y)
        y = self.block2_conv2(y)
        y = self.block2_pool(y)

        y = self.block3_conv1(y)
        y = self.block3_conv2(y)
        y = self.block3_conv3(y)
        y = self.block3_pool(y)

        y = self.block4_conv1(y)
        y = self.block4_conv2(y)
        y = self.block4_conv3(y)
        y = self.block4_pool(y)

        y = self.block5_conv1(y)
        y = self.block5_conv2(y)
        y = self.block5_conv3(y)

        return y
