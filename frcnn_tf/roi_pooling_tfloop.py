import tensorflow as tf
import numpy as np

from mini_batch_tf import tf_loop_fn, filter_tensors, IoU_tf

class RoIPooling(tf.keras.layers.Layer):
    
    def __init__(self):
        super().__init__()

    def pool_roi(self,roi, features):

        feature_map=tf.squeeze(features)
        feature_map_height = int(feature_map.shape[0])
        feature_map_width  = int(feature_map.shape[1])
        
        h_start = tf.cast(feature_map_height * roi[0], 'int32')
        w_start = tf.cast(feature_map_width  * roi[1], 'int32')
        h_end   = tf.cast(feature_map_height * roi[2], 'int32')
        w_end   = tf.cast(feature_map_width  * roi[3], 'int32')
        
        region = feature_map[h_start:h_end, w_start:w_end, :]

        region_height = h_end - h_start
        region_width  = w_end - w_start
        
        h_step = tf.cast( region_height / 7, 'int32')
        w_step = tf.cast( region_width  / 7, 'int32')

        areas = [[(i*h_step, j*w_step, (i+1)*h_step if i+1 < 7 else region_height, (j+1)*w_step if j+1 < 7 else region_width) for j in range(7)] for i in range(7)]

        def pool_area(x):
                corners=list(x)
                if (corners[0]==corners[2])&(corners[1]==corners[3]):
                    if corners[2]<feature_map_height:
                        i=corners[2]
                        v=tf.cast(1, 'int32')
                        corners[2]=tf.add(i, v)
                    else:
                        i=corners[0]
                        v=tf.cast(-1, 'int32')
                        corners[0]=tf.add(i, v)
                    if corners[3]<feature_map_width:
                            i=corners[3]
                            v=tf.cast(1, 'int32')
                            corners[3]=tf.add(i, v)
                    else:
                            i=corners[1]
                            v=tf.cast(-1, 'int32')
                            corners[1]=tf.add(i, v)
                    x=corners
                    return tf.math.reduce_max(region[x[0]:x[2], x[1]:x[3], :], axis=[0,1])
                elif corners[0]==corners[2]&corners[1]!=corners[3]:
                    if corners[2]<feature_map_height:
                        i=corners[2]
                        v=tf.cast(1, 'int32')
                        corners[2]=tf.add(i, v)
                    else:
                        i=corners[0]
                        v=tf.cast(-1, 'int32')
                        corners[0]=tf.add(i, v)
                    x=corners
                    return tf.math.reduce_max(region[x[0]:x[2], x[1]:x[3], :], axis=[0,1])
                elif corners[1]==corners[3]&corners[0]!=corners[2]:
                    if corners[3]<feature_map_width:
                            i=corners[3]
                            v=tf.cast(1, 'int32')
                            corners[3]=tf.add(i, v)
                    else:
                            i=corners[1]
                            v=tf.cast(-1, 'int32')
                            corners[1]=tf.add(i, v)
                    x=corners
                    return tf.math.reduce_max(region[x[0]:x[2], x[1]:x[3], :], axis=[0,1])
                else:
                    x=corners
                    return tf.math.reduce_max(region[x[0]:x[2], x[1]:x[3], :], axis=[0,1]) 
        
        pooled_features = tf.stack([[pool_area(x) for x in row] for row in areas])
        return pooled_features

    def multiple_roi(self, features, cls, bbox):
        
        size=len(cls[0])

        x= tf_loop_fn(cls[0], bbox[0], size)

        def helper(x):
            return self.pool_roi(x, features)
        
        def tf_loop_fn_1(x, size):

            def body_fn(iteration, tensor_array):
                new_tensor = helper(tf.gather(x, indices=iteration, axis=0))
                tensor_array = tensor_array.write(iteration, new_tensor)
                iteration = tf.add(iteration, 1)
                return iteration, tensor_array

            def cond_fn(iteration, tensor_array):
                return tf.less(iteration, size)

            tensor_array = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

            iteration = tf.constant(0)

            _, tensor_array = tf.while_loop(cond_fn, body_fn, loop_vars=[iteration, tensor_array])

            return tensor_array.stack()

        pooled_areas=tf_loop_fn_1(x, size)
        
        def create_tensor_list(x, n):
            counter = tf.constant(0)
            tensor_list = tf.TensorArray(tf.float32, size=n)

            def loop_condition(counter, tensor_list):
                return tf.less(counter, n)

            def loop_body(counter, tensor_list):
                tensor=IoU_tf(tf.gather(x, indices=counter, axis=0))
                tensor_list = tensor_list.write(counter, tensor)
                return counter + 1, tensor_list

            _, tensor_list = tf.while_loop(loop_condition, loop_body, [counter, tensor_list])
            
            tensor_list = tensor_list.stack()

            return tensor_list
        
        iou=create_tensor_list(x, size)
        #size_1=tf.constant(1)
        #print(iou)
        iou_filtered, filtered_roi, cls_filtered, bbox_filtered = filter_tensors(iou, pooled_areas, cls, bbox)
        #print(iou_filtered)
        
        return iou_filtered, filtered_roi, cls_filtered, bbox_filtered
        
    def all_roi(self, features, cls, bbox):
        x=[features, cls, bbox]

        def helper2(x):
            return self.multiple_roi(x[0], x[1], x[2])
        
        all_dataset=helper2(x)
        
        return all_dataset