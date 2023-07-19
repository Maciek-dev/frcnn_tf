import tensorflow as tf
from keras import backend as K

from batch_dataset import DataLoader
from feature_extractor import FeatureExtractor
from roi_pooling_tfloop import RoIPooling
from mini_batch_tf import mini_batch_tf
from detectron_tf import Detectron
from tensorflow.python.framework import ops

class FastRCNN(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.features=FeatureExtractor(l2=0.01)
        self.all_dataset=RoIPooling()
        self.detectron=Detectron()
    
    def call(self, dataset):

        features=self.features.call(dataset[0])
        
        all_dataset=self.all_dataset.all_roi(features, dataset[1][1], dataset[1][0])

        mini=mini_batch_tf(all_dataset[0], all_dataset[1], all_dataset[2], all_dataset[3])
        print(mini[0])
        def tf_pred_loop(mini):
            
            size=mini[1].shape[0]
            #print(size)

            def body_fn(iteration, tensor_cls, tensor_bbox):
                roi=mini[1][iteration]
                pred_cls, pred_bbox = self.detectron.call(roi)
                tensor_cls = tensor_cls.write(iteration, pred_cls)
                tensor_bbox =tensor_bbox.write(iteration, pred_bbox)                
                iteration = tf.add(iteration, 1)
                return iteration, tensor_cls, tensor_bbox

            def cond_fn(iteration, tensor_cls, tensor_bbox):
                return tf.less(iteration, size)

            tensor_cls = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
            tensor_bbox = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
            
            iteration = tf.constant(0)
            _, tensor_cls, tensor_bbox = tf.while_loop(cond_fn, body_fn, loop_vars=[iteration, tensor_cls, tensor_bbox])

            return tensor_cls.stack(), tensor_bbox.stack()

        #if mini[1].shape[0]!=None:
        #    pred_cls, pred_bbox=tf_pred_loop(mini)
        #    return pred_cls, pred_bbox

        def tf_loss(pred_cls, pred_bbox , mini):
            
            size=mini[2].shape[0]
            #print(size)

            def body_fn(iteration, tensor_cls_loss, tensor_reg_loss):
                tensor_cls_loss=self.detectron.cls_results(mini[2][iteration], pred_cls)
                tensor_reg_loss=self.detectron.reg_results(mini[2][iteration], mini[3][iteration], pred_bbox[iteration])
                #tensor_cls_loss = tensor_cls_loss.write(iteration, cls_loss)
                #tensor_reg_loss =tensor_reg_loss.write(iteration, reg_loss)                
                iteration = tf.add(iteration, 1)
                return iteration, tensor_cls_loss, tensor_reg_loss

            def cond_fn(iteration, tensor_cls_loss, tensor_reg_loss):
                return tf.less(iteration, size)

            tensor_cls_loss = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
            tensor_reg_loss = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
            
            iteration = tf.constant(0)
            _, tensor_cls_loss, tensor_reg_loss = tf.while_loop(cond_fn, body_fn, loop_vars=[iteration, tensor_cls_loss, tensor_reg_loss])
            
            #print(tensor_cls_loss.stack())
            #print(tensor_reg_loss.stack())
            #tensor_cls_loss=tensor_cls_loss.stack()
            #tensor_reg_loss=tensor_reg_loss.stack()
            loss=tf.math.add(tensor_cls_loss, tensor_reg_loss)
            
            return loss
        
        #if mini[1].shape[0]!=None:
        pred_cls, pred_bbox=tf_pred_loop(mini)
            #return pred_cls, pred_bbox
        loss=tf_loss(pred_cls, pred_bbox ,mini)
        #    return loss
        #else:
        #    loss=tf.constant(0, dtype=tf.float32)
        #    return loss

        return loss


input_dir = "/home/maciek/Documents/images/schematic/img/"
target_dir = "/home/maciek/Documents/images/schematic/ann/"

train=DataLoader(input_dir, target_dir, True)
val=DataLoader(input_dir, target_dir, False)

model=FastRCNN()

optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, weight_decay=0.01)

@tf.function
def train_step(data):
    with tf.GradientTape() as tape:
        loss=model.call(data)
        #ops.reset_default_graph()
    #print(loss)
    grads = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
            #train_acc_metric.update_state(y, logits)
    #tf.reset_default_graph()
    return loss

@tf.function
def test_step(data):
    val_loss = model(data, training=False)
    #tf.reset_default_graph()

    return val_loss

epochs = 1
for epoch in range(epochs):
    tf.print("\nStart of epoch %d" % (epoch,))

    for step, data in enumerate(train):

        loss=train_step(data)
        #tf.reset_default_graph()

        if step % 10 == 0:
                tf.print(
                    "Training loss (for one batch) at step %d: %.2f"
                    % (step, float(loss))
                #    "Training loss (for one batch) at step %d"
                #    % (step)
                )
                tf.print("Seen so far: %s samples" % ((step + 1)))

    for val_step, val_data in enumerate(val):

        val_loss=test_step(val_data)
        
        if val_step % 10 == 0:
                tf.print(
                    "Validation loss (for one batch) at step %d: %.2f"
                    % (val_step, float(val_loss))
                )
                tf.print("Seen so far: %s samples" % ((val_step + 1)))

