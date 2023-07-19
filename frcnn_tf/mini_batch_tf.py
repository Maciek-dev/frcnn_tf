import tensorflow as tf

tf.random.set_seed(42)

def extract_roi(cls, bbox, line=False):
        a=0
        for cl in cls:
            if cl!=0:
                break     
            a+=1
        x=bbox[a]
        if line==False:
            return x
        elif line==True:
            return x, a

def tf_loop_fn(cls, bbox, size):

    def body_fn(iteration, tensor_array):
        new_tensor = extract_roi(cls[iteration], bbox[iteration])
        tensor_array = tensor_array.write(iteration, new_tensor)
        iteration = tf.add(iteration, 1)
        return iteration, tensor_array

    def cond_fn(iteration, tensor_array):
        return tf.less(iteration, size)

    tensor_array = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

    iteration = tf.constant(0)
    _, tensor_array = tf.while_loop(cond_fn, body_fn, loop_vars=[iteration, tensor_array])

    return tensor_array.stack()
'''
def filter_tensors(iou, pooled_areas, cls, bbox):
    #print(iou.get_shape()[0])
    #i = tf.constant(0)
    #size = iou.get_shape()[0]
    #print(tf.shape(iou)[0])
    #iou_filtered=tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    #filtered_roi = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    #cls_filtered = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    #bbox_filtered = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    
    def condition(i, iou_filtered, filtered_roi, cls_filtered, bbox_filtered):
        return tf.less(i, tf.shape(iou)[0])

    def body(i, iou_filtered, filtered_roi, cls_filtered, bbox_filtered):
        
        a = pooled_areas[i]
        max_value = tf.reduce_max(a)

        def true_fn(iou_filtered, filtered_roi, cls_filtered, bbox_filtered):
            iou_filtered=iou_filtered.write(iou_filtered.size(), iou[i])
            filtered_roi = filtered_roi.write(filtered_roi.size(), tf.cast(a, dtype=tf.float32))
            cls_filtered = cls_filtered.write(cls_filtered.size(), cls[0][i])
            bbox_filtered = bbox_filtered.write(bbox_filtered.size(), bbox[0][i])
            return iou_filtered, filtered_roi, cls_filtered, bbox_filtered

        iou_filtered, filtered_roi, cls_filtered, bbox_filtered = tf.cond(max_value >= 0, 
                                                                          lambda: true_fn(iou_filtered, filtered_roi, cls_filtered, bbox_filtered) ,
                                                                          lambda: (iou_filtered, filtered_roi, cls_filtered, bbox_filtered))
        return i + 1, iou_filtered, filtered_roi, cls_filtered, bbox_filtered

    i = tf.constant(0)

    iou_filtered=tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    filtered_roi = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    cls_filtered = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    bbox_filtered = tf.TensorArray(tf.float32, size=0, dynamic_size=True)

    _, iou_filtered, filtered_roi, cls_filtered, bbox_filtered = tf.while_loop(condition, body,[i, iou_filtered, filtered_roi, cls_filtered, bbox_filtered])

    iou_filtered = iou_filtered.stack()
    filtered_roi = filtered_roi.stack()
    cls_filtered = cls_filtered.stack()
    bbox_filtered = bbox_filtered.stack()
    print(iou_filtered)
    #print(filtered_roi)
    #print(cls_filtered)
    #print(bbox_filtered)
    return iou_filtered, filtered_roi, cls_filtered, bbox_filtered
'''
def filter_tensors(iou, pooled_areas, cls, bbox):
    def condition(i, iou_filtered, filtered_roi, cls_filtered, bbox_filtered):
        return tf.less(i, tf.shape(iou)[0])

    def body(i, iou_filtered, filtered_roi, cls_filtered, bbox_filtered):
        a = pooled_areas[i]
        max_value = tf.reduce_max(a)

        def true_fn(iou_filtered, filtered_roi, cls_filtered, bbox_filtered):
            iou_filtered = iou_filtered.write(i, iou[i])
            filtered_roi = filtered_roi.write(i, tf.cast(a, dtype=tf.float32))
            cls_filtered = cls_filtered.write(i, cls[0][i])
            bbox_filtered = bbox_filtered.write(i, bbox[0][i])
            return iou_filtered, filtered_roi, cls_filtered, bbox_filtered

        iou_filtered, filtered_roi, cls_filtered, bbox_filtered = tf.cond(
            max_value >= 0, 
            lambda: true_fn(iou_filtered, filtered_roi, cls_filtered, bbox_filtered),
            lambda: (iou_filtered, filtered_roi, cls_filtered, bbox_filtered)
        )

        return i + 1, iou_filtered, filtered_roi, cls_filtered, bbox_filtered

    i = tf.constant(0)
    size = tf.shape(iou)[0]

    iou_filtered = tf.TensorArray(tf.float32, size=size, dynamic_size=True)
    filtered_roi = tf.TensorArray(tf.float32, size=size, dynamic_size=True)
    cls_filtered = tf.TensorArray(tf.float32, size=size, dynamic_size=True)
    bbox_filtered = tf.TensorArray(tf.float32, size=size, dynamic_size=True)

    _, iou_filtered, filtered_roi, cls_filtered, bbox_filtered = tf.while_loop(
        condition, body, loop_vars=[i, iou_filtered, filtered_roi, cls_filtered, bbox_filtered]
    )

    iou_filtered = iou_filtered.stack()
    filtered_roi = filtered_roi.stack()
    cls_filtered = cls_filtered.stack()
    bbox_filtered = bbox_filtered.stack()

    return iou_filtered, filtered_roi, cls_filtered, bbox_filtered





def IoU_tf(roi):    
    a=[tf.cast(roi[0]*512, 'int32'), tf.cast(roi[1]*512, 'int32'),tf.cast(roi[2]*512, 'int32'),tf.cast(roi[3]*512, 'int32')]
    h_start = tf.cast(32 * roi[0], 'int32')*16
    w_start = tf.cast(32 * roi[1], 'int32')*16
    h_end   = tf.cast(32 * roi[2], 'int32')*16
    w_end   = tf.cast(32 * roi[3], 'int32')*16

    c1=tf.maximum(a[0], h_start)
    c2=tf.maximum(a[1], w_start)
    c3=tf.minimum(a[2], h_end)
    c4=tf.minimum(a[3], w_end)

    intersection=(c3-c1)*(c4-c2)
    f1=(a[3]-a[1])*(a[2]-a[0])
    f2=(w_end-w_start)*(h_end-h_start)
    union=f1+f2-intersection
    IoU=intersection/union
    IoU=tf.cast(IoU, 'float32')

    return IoU

def pos_batch(iou, roi_list, bbox_list, cls_list):
    
    i = tf.constant(0)
    
    iou_pos=tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    roi_pos = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    cls_pos = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    bbox_pos = tf.TensorArray(tf.float32, size=0, dynamic_size=True)

    def loop_condition(i, iou_pos, roi_pos, bbox_pos, cls_pos):
        return tf.less(i, tf.shape(iou)[0])

    def loop_body(i, iou_pos, roi_pos, bbox_pos, cls_pos):

        def append_pos(iou_pos, roi_pos, bbox_pos, cls_pos):
            iou_pos=iou_pos.write(iou_pos.size(), iou[i])
            roi_pos=roi_pos.write(roi_pos.size(), roi_list[i])
            bbox_pos=bbox_pos.write(bbox_pos.size(), bbox_list[i])
            cls_pos=cls_pos.write(cls_pos.size(), cls_list[i])
            return iou_pos, roi_pos, bbox_pos, cls_pos

        tf.cond(tf.greater(iou[i], 0.5), lambda:append_pos(iou_pos, roi_pos, bbox_pos, cls_pos), lambda: (iou_pos, roi_pos, bbox_pos, cls_pos))

        return i + 1, iou_pos, roi_pos, bbox_pos, cls_pos
    
    _, iou_pos, roi_pos, bbox_pos, cls_pos= tf.while_loop(loop_condition, loop_body, loop_vars=[i, iou_pos, roi_pos, bbox_pos, cls_pos])

    iou_pos=iou_pos.stack()
    roi_pos=roi_pos.stack()
    bbox_pos=bbox_pos.stack()
    cls_pos=cls_pos.stack()

    return iou_pos, roi_pos, bbox_pos, cls_pos

def neg_batch(iou, roi_list, bbox_list, cls_list):
    
    j = tf.constant(0)
    
    iou_neg=tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    roi_neg = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    cls_neg = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    bbox_neg = tf.TensorArray(tf.float32, size=0, dynamic_size=True)

    def loop_condition_1(j, iou_neg, roi_neg, bbox_neg, cls_neg):
        return tf.less(j, tf.shape(iou)[0])

    def loop_body_1(j, iou_neg, roi_neg, bbox_neg, cls_neg):

        def append_neg(iou_neg, roi_neg, bbox_neg, cls_neg):
            iou_neg=iou_neg.write(iou_neg.size(), iou[j])
            roi_neg=roi_neg.write(roi_neg.size(), roi_list[j])
            bbox_neg=bbox_neg.write(bbox_neg.size(), bbox_list[j])
            cls_neg=cls_neg.write(cls_neg.size(), cls_list[j])
            return iou_neg, roi_neg, bbox_neg, cls_neg

        tf.cond(tf.logical_and(tf.greater(iou[j], 0.1), tf.less_equal(iou[j], 0.5)), lambda:append_neg(iou_neg, roi_neg, bbox_neg, cls_neg), lambda: (iou_neg, roi_neg, bbox_neg, cls_neg))

        return j + 1, iou_neg, roi_neg, bbox_neg, cls_neg
    
    _, iou_neg, roi_neg, bbox_neg, cls_neg = tf.while_loop(loop_condition_1, loop_body_1, loop_vars=[j, iou_neg, roi_neg, bbox_neg, cls_neg])

    iou_neg=iou_neg.stack()
    roi_neg=roi_neg.stack()
    bbox_neg=bbox_neg.stack()
    cls_neg=cls_neg.stack()

    return iou_neg, roi_neg, bbox_neg, cls_neg

def mini_batch_tf(iou, roi_list, bbox_list, cls_list):
    #print(f'iou: {iou}')
    iou_pos, roi_pos, bbox_pos, cls_pos =pos_batch(iou, roi_list, bbox_list, cls_list)
    iou_neg, roi_neg, bbox_neg, cls_neg =neg_batch(iou, roi_list, bbox_list, cls_list)
    
    def num_samples(stack, n):
        x=tf.constant(n)
        y=tf.shape(stack)
        y=tf.reshape(y, x.shape)
        return x-y

    def shuffle_stack(stack, num_samples):
        shuffled_stack=tf.random.shuffle(stack)
        sampled_stack=shuffled_stack[:num_samples]
        return sampled_stack

    if tf.equal(tf.shape(iou_pos),0) and tf.greater(tf.shape(iou_neg),0):
        if tf.less(tf.shape(iou_neg), 64):
            
            num=num_samples(iou_neg, 64)

            sample_iou=shuffle_stack(iou_neg, num)
            sample_roi=shuffle_stack(roi_neg, num)
            sample_bbox=shuffle_stack(bbox_neg, num)
            sample_cls=shuffle_stack(cls_neg, num)
            
            mini_iou = tf.concat([iou_neg, sample_iou], axis=0)
            mini_roi = tf.concat([roi_neg, sample_roi], axis=0)
            mini_bbox = tf.concat([bbox_neg, sample_bbox], axis=0)
            mini_cls = tf.concat([cls_neg, sample_cls], axis=0)

            mini_batch = [mini_iou, mini_roi, mini_bbox, mini_cls]
            
        else:
            mini_batch = [iou_neg[:64], roi_neg[:64], bbox_neg[:64], cls_neg[:64]]
    
    elif tf.equal(tf.shape(iou_neg), 0) and tf.greater(tf.shape(iou_pos), 0):
        if tf.less(tf.shape(iou_pos), 64):
            
            num=num_samples(iou_pos, 64)
            
            sample_iou=shuffle_stack(iou_pos, num)
            sample_roi=shuffle_stack(roi_pos, num)
            sample_bbox=shuffle_stack(bbox_pos, num)
            sample_cls=shuffle_stack(cls_pos, num)

            mini_iou = tf.concat([iou_pos, sample_iou], axis=0)
            mini_roi = tf.concat([roi_pos, sample_roi], axis=0)
            mini_bbox = tf.concat([bbox_pos, sample_bbox], axis=0)
            mini_cls = tf.concat([cls_pos, sample_cls], axis=0)

            mini_batch = [mini_iou, mini_roi, mini_bbox, mini_cls]
        else:
            mini_batch = [iou_pos[:64], roi_pos[:64], bbox_pos[:64], cls_pos[:64]]
    
    elif tf.logical_and(tf.less(tf.shape(iou_pos), 16), tf.less(tf.shape(iou_neg), 48)):
        num_pos=num_samples(iou_pos, 16)
        num_neg=num_samples(iou_neg, 48)

        sample_pos_iou=shuffle_stack(iou_pos, num_pos)
        sample_pos_roi=shuffle_stack(roi_pos, num_pos)
        sample_pos_bbox=shuffle_stack(bbox_pos, num_pos)
        sample_pos_cls=shuffle_stack(cls_pos, num_pos)

        sample_neg_iou=shuffle_stack(iou_neg, num_neg)
        sample_neg_roi=shuffle_stack(roi_neg, num_neg)
        sample_neg_bbox=shuffle_stack(bbox_neg, num_neg)
        sample_neg_cls=shuffle_stack(cls_neg, num_neg)

        mini_pos_iou = tf.concat([iou_pos, sample_pos_iou], axis=0)
        mini_pos_roi = tf.concat([roi_pos, sample_pos_roi], axis=0)
        mini_pos_bbox = tf.concat([bbox_pos, sample_pos_bbox], axis=0)
        mini_pos_cls = tf.concat([cls_pos, sample_pos_cls], axis=0)

        mini_neg_iou = tf.concat([iou_neg, sample_neg_iou], axis=0)
        mini_neg_roi = tf.concat([roi_neg, sample_neg_roi], axis=0)
        mini_neg_bbox = tf.concat([bbox_neg, sample_neg_bbox], axis=0)
        mini_neg_cls = tf.concat([cls_neg, sample_neg_cls], axis=0)
        
        mini_iou = tf.concat([mini_pos_iou, mini_neg_iou], axis=0)
        mini_roi = tf.concat([mini_pos_roi, mini_neg_roi], axis=0)
        mini_bbox = tf.concat([mini_pos_bbox, mini_neg_bbox], axis=0)
        mini_cls = tf.concat([mini_pos_cls, mini_neg_cls], axis=0)
        
        mini_batch = [mini_iou, mini_roi, mini_bbox, mini_cls]
    
    elif tf.logical_and(tf.less(tf.shape(iou_pos), 16), tf.greater_equal(tf.shape(iou_neg), 48)):
        num_pos=num_samples(iou_pos, 16)

        sample_pos_iou=shuffle_stack(iou_pos, num_pos)
        sample_pos_roi=shuffle_stack(roi_pos, num_pos)
        sample_pos_bbox=shuffle_stack(bbox_pos, num_pos)
        sample_pos_cls=shuffle_stack(cls_pos, num_pos)
        
        mini_pos_iou = tf.concat([iou_pos, sample_pos_iou], axis=0)
        mini_pos_roi = tf.concat([roi_pos, sample_pos_roi], axis=0)
        mini_pos_bbox = tf.concat([bbox_pos, sample_pos_bbox], axis=0)
        mini_pos_cls = tf.concat([cls_pos, sample_pos_cls], axis=0)

        mini_neg_iou = [iou_neg[:48]]
        mini_neg_roi = [roi_neg[:48]]
        mini_neg_bbox = [bbox_neg[:48]]
        mini_neg_cls = [cls_neg[:48]]
        
        mini_iou = tf.concat([mini_pos_iou, mini_neg_iou[0]], axis=0)
        mini_roi = tf.concat([mini_pos_roi, mini_neg_roi[0]], axis=0)
        mini_bbox = tf.concat([mini_pos_bbox, mini_neg_bbox[0]], axis=0)
        mini_cls = tf.concat([mini_pos_cls, mini_neg_cls[0]], axis=0)
        
        mini_batch = [mini_iou, mini_roi, mini_bbox, mini_cls]
    
    elif tf.logical_and(tf.greater_equal(tf.shape(iou_pos), 16), tf.less(tf.shape(iou_neg), 48)):
        num_neg=num_samples(iou_neg, 48)
        
        sample_neg_iou=shuffle_stack(iou_neg, num_neg)
        sample_neg_roi=shuffle_stack(roi_neg, num_neg)
        sample_neg_bbox=shuffle_stack(bbox_neg, num_neg)
        sample_neg_cls=shuffle_stack(cls_neg, num_neg)
        
        mini_neg_iou = tf.concat([iou_neg, sample_neg_iou], axis=0)
        mini_neg_roi = tf.concat([roi_neg, sample_neg_roi], axis=0)
        mini_neg_bbox = tf.concat([bbox_neg, sample_neg_bbox], axis=0)
        mini_neg_cls = tf.concat([cls_neg, sample_neg_cls], axis=0)

        mini_pos_iou = [iou_pos[:16]]
        mini_pos_roi = [roi_pos[:16]]
        mini_pos_bbox = [bbox_pos[:16]]
        mini_pos_cls = [cls_pos[:16]]

        mini_iou = tf.concat([mini_pos_iou[0], mini_neg_iou], axis=0)
        mini_roi = tf.concat([mini_pos_roi[0], mini_neg_roi], axis=0)
        mini_bbox = tf.concat([mini_pos_bbox[0], mini_neg_bbox], axis=0)
        mini_cls = tf.concat([mini_pos_cls[0], mini_neg_cls], axis=0)
        
        mini_batch = [mini_iou, mini_roi, mini_bbox, mini_cls]
    
    elif tf.logical_and(tf.greater_equal(tf.shape(iou_pos), 16), tf.greater_equal(tf.shape(iou_neg), 48)):
    #else:    
        mini_neg_iou = [iou_neg[:48]]
        mini_neg_roi = [roi_neg[:48]]
        mini_neg_bbox = [bbox_neg[:48]]
        mini_neg_cls = [cls_neg[:48]]
                
        mini_pos_iou = [iou_pos[:16]]
        mini_pos_roi = [roi_pos[:16]]
        mini_pos_bbox = [bbox_pos[:16]]
        mini_pos_cls = [cls_pos[:16]]

        mini_iou = tf.concat([mini_pos_iou[0], mini_neg_iou[0]], axis=0)
        mini_roi = tf.concat([mini_pos_roi[0], mini_neg_roi[0]], axis=0)
        mini_bbox = tf.concat([mini_pos_bbox[0], mini_neg_bbox[0]], axis=0)
        mini_cls = tf.concat([mini_pos_cls[0], mini_neg_cls[0]], axis=0)
        
        mini_batch = [mini_iou, mini_roi, mini_bbox, mini_cls]

    #else:
        #mini_batch=[tf.constant(0, dtype=tf.float32), tf.constant(0, dtype=tf.float32), tf.constant(0, dtype=tf.float32), tf.constant(0, dtype=tf.float32)]

    return mini_batch