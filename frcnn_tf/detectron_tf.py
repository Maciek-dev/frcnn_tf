import tensorflow as tf
from keras.layers import Dense, Dropout, Flatten
import numpy as np
from keras import backend as K

from mini_batch_tf import extract_roi

class Detectron(tf.keras.Model):
    
    def __init__(self):
        super().__init__()

        regularizer = tf.keras.regularizers.l2(0.01)
        class_initializer = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = 0.01)
        regressor_initializer = tf.keras.initializers.RandomNormal(mean = 0.0, stddev = 0.001)

        self.flatten=Flatten(name='flatten')
        self.dense1 = Dense(name='dense1',units=4096, activation='relu', kernel_regularizer=regularizer)
        self.dropout1=Dropout(name='drop1', rate=0.5)
        self.dense2 = Dense(name='dense2',units=4096, activation='relu', kernel_regularizer=regularizer)
        self.dropout2=Dropout(name='drop2', rate=0.5)

        self.cls_pred=Dense(name='cls_pred', units=9, activation='softmax', kernel_regularizer=class_initializer)
        
        self.reg_pred=Dense(name='reg_pred', units=32, activation='linear', kernel_regularizer=regressor_initializer)

    def call(self, roi_list):
        roi_list=tf.expand_dims(roi_list, axis=0)
        y=self.flatten(roi_list)
        y = self.dense1(y)
        y=self.dropout1(y)
        y = self.dense2(y)
        y=self.dropout2(y)

        cls_pred=self.cls_pred(y)
        
        reg_pred=self.reg_pred(y)

        return cls_pred, reg_pred
    
    def cls_results(self, cls_list, cls_pred):
        zero=tf.constant(0, shape=(1,), dtype=tf.float32)
        x=tf.concat([zero, cls_list], axis=0)
        a=cls_pred[0][0]
        cls_loss= K.sum(K.categorical_crossentropy(target=x, output=a, from_logits=False))
        return cls_loss

    def reg_results(self, cls_list, bbox_list, reg_pred):
        y_true, b =extract_roi(cls_list, bbox_list, True)
        y_pred=tf.cast([reg_pred[0][((b+1)*4)-4],reg_pred[0][((b+1)*4)-3],reg_pred[0][((b+1)*4)-2],reg_pred[0][((b+1)*4)-1]], dtype=tf.float32)
        z=y_true-y_pred
        z_abs=tf.math.abs(z)
        
        '''
        losses=[]
        for ind,r in enumerate(z_abs):
            if r<1.0:
                loss=0.5*z[ind]*z[ind]*1.0
                losses.append(loss)
            else:
                loss=r-0.5 / 1.0
                losses.append(loss)
        reg_losses=K.sum(tf.cast(losses, dtype=tf.float32))
        '''

        def regression(z_abs, z):

            i = tf.constant(0)
            size = z_abs.shape[0]

            losses=tf.TensorArray(tf.float32, size=0, dynamic_size=True)
            
            def condition(i, losses):
                return i<size

            def body(i, losses):
                
                a = z_abs[i]

                def true_fn(losses):
                    loss=0.5*z[i]*z[i]*1.0
                    losses=losses.write(losses.size(), tf.cast(loss, dtype=tf.float32))
                    return losses
                
                def false_fn(losses):
                    loss=a-0.5 / 1.0
                    losses=losses.write(losses.size(), tf.cast(loss, dtype=tf.float32))
                    return losses

                losses = tf.cond(a<1.0, lambda: true_fn(losses),lambda: false_fn(losses))
                return i + 1, losses

            _, losses = tf.while_loop(condition, body,[i, losses])

            losses = losses.stack()

            return losses

        reg_losses=K.sum(regression(z_abs, z))

        return reg_losses

