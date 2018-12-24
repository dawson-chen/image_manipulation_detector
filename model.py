#!/usr/bin/env python
# coding: utf-8

# # A Deep Learning Approach To Universal Image Manipulation Detection Using A New Convolutional Layer

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import math
import sys
from glob import glob
import cv2


# In[ ]:


from tqdm import tqdm 


# In[ ]:


def images_square_grid(images):
    """
    Save images as a square grid
    :param images: Images to be used for the grid
    :param mode: The mode to use for images
    :return: Image of images in a square grid
    """
    # Get maximum size for square grid of images
    save_size = math.floor(np.sqrt(images.shape[0]))

    # Scale to 0-255
    images = (((images - images.min()) * 255) / (images.max() - images.min())).astype(np.uint8)

    # Put images in a square arrangement
    images_in_square = np.reshape(
            images[:save_size*save_size],
            (save_size, save_size, images.shape[1], images.shape[2]))
    h = images.shape[1]
    w = images.shape[2]
    # Combine images to grid image
    new_im = np.ones((save_size*h, save_size*w), dtype=np.uint8)
    for col_i, col_images in enumerate(images_in_square):
        for image_i, image in enumerate(col_images):
            new_im[col_i * h: col_i * h + h, 
                   image_i * h: image_i * h + h] = image

    return new_im


# In[ ]:





# In[ ]:


IMAGE_SIZE = 64
IMAGE_CHANNEL = 1
NUM_LABELS = 2

CONV_RES_DEEP = 12
CONV_RES_SIZE = 5

CONV1_DEEP = 64
CONV1_SIZE = 7

CONV2_DEEP = 48
CONV2_SIZE = 5

FC_SIZE1 = 256
FC_SIZE2 = 256


# In[ ]:


# generate data using a batch size 64
# where input shape is (64, 227, 227, 1)
def parser(record):
    features = tf.parse_single_example(
        record,
        features={
            "image_raw" : tf.FixedLenFeature([], tf.string),
            "label" : tf.FixedLenFeature([], tf.int64)
        }
    )
    image = tf.decode_raw(features['image_raw'], tf.uint8)
#     image.set_shape([256, 256])
    image = tf.reshape(image, [256, 256, 1])
    label = features['label']
    
    return image, label

def preprocessing(image, label):
    img = tf.image.resize_image_with_crop_or_pad(image, IMAGE_SIZE, IMAGE_SIZE)
    img = tf.image.convert_image_dtype(img, tf.float32)
    # scale image to 0~1
#     img = (((img - tf.reduce_min(img))) / (tf.reduce_max(img) - tf.reduce_min(img)))
    img = img / 255
    
    ont_hot = tf.one_hot(label, depth=NUM_LABELS)
#     label = tf.expand_dims(label , -1)
    
    return img, ont_hot

def dataset(file, batch_size=32, 
            num_epochs=1, is_shuffle=False, shuffle_buffer=10000, 
            preprocess=preprocessing):
#     if train_file is None:
#         train_file = "mini_dataset/my_dataset/train.tfrecords"
#     if test_file is None:
#         test_file = "mini_dataset/my_dataset/test.tfrecords"
    input_file = tf.train.match_filenames_once(file)
    dataset = tf.data.TFRecordDataset(input_file)
    dataset = dataset.map(parser)
    dataset = dataset.map(preprocess)
    if is_shuffle:
        dataset = dataset.shuffle(shuffle_buffer)
    
    dataset = dataset.batch(batch_size)

    dataset = dataset.repeat(num_epochs)
    
    return dataset


# In[ ]:


# test dataset and preprocessing
data = dataset("tfrecords/MF_clf_train.tfrecords-*")
ite = data.make_initializable_iterator()
img_batch, label_batch = ite.get_next()
with tf.Session() as sess:
    sess.run([tf.global_variables_initializer(),
              tf.local_variables_initializer()])
    sess.run(ite.initializer)
    for i in range(2):
        image, lab = sess.run([img_batch, label_batch])
        print("    ", image.shape, image.dtype)
        print("    ", lab.shape, lab.dtype)
        
plt.figure(figsize=(8,8))
plt.imshow(images_square_grid(image), cmap="gray");


# In[ ]:


def get_placeholder():
    input_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name='input_placeholder')
    
    labels_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, NUM_LABELS], name='label_placeholder')
    
    dropout_placeholder = tf.placeholder(dtype=tf.float32, shape=[], name='dropout_placeholder')
    
    return input_placeholder, labels_placeholder, dropout_placeholder


# In[ ]:


def constraint_weights(weights):
#     assert weights.shape == (CONV_RES_SIZE, CONV_RES_SIZE, IMAGE_CHANNEL, CONV_RES_DEEP)
    mid_inx = CONV_RES_SIZE // 2
    weights[mid_inx, mid_inx, :, :] = 0
    weights = weights / np.sum(weights, axis=(0, 1))
    weights[mid_inx, mid_inx, :, :] = -1
    
    return weights


# In[ ]:


def get_weights(shape, name="weights"):
    return tf.Variable(tf.truncated_normal(shape=shape, stddev=0.1), name=name)

def get_bias(shape, name="bias"):
    return tf.Variable(tf.constant(0.01, shape=shape), dtype=tf.float32, name = name)


# In[ ]:


def build_nn(input_tensor, labels, dropout):
    with tf.variable_scope('layer1_conv_res'):
        conv_res_weights = get_weights([CONV_RES_SIZE, CONV_RES_SIZE, IMAGE_CHANNEL, CONV_RES_DEEP])
        conv_res_bias = get_bias([CONV_RES_DEEP, ])
        
        conv = tf.nn.conv2d(input_tensor, conv_res_weights, strides=[1,1,1,1], padding='VALID')
        layer1 = tf.nn.bias_add(conv, conv_res_bias)
        
        print("conv1", layer1.get_shape().as_list())
    # BATHCH_SIZE, 223, 223, 12
    with tf.variable_scope('layer2_conv1'):
        conv1_wights = get_weights([CONV1_SIZE, CONV1_SIZE, CONV_RES_DEEP, CONV1_DEEP])
        conv1_bias = get_bias([CONV1_DEEP, ])
        
        conv1 = tf.nn.conv2d(layer1, conv1_wights, strides=[1, 2, 2, 1], padding='SAME')
        conv1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_bias))
        conv1 = tf.nn.lrn(conv1, depth_radius=5, bias=2, alpha=1e-4, beta=.75)
        layer2 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
        
        print("conv2", layer2.get_shape().as_list())
    # batch_size, 56, 56, 64
    with tf.variable_scope('layer3_conv2'):
        conv2_wights = get_weights([CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP])
        conv2_bias = get_bias([CONV2_DEEP, ])
        
        conv2 = tf.nn.conv2d(layer2, conv2_wights, strides=[1, 1, 1, 1], padding='SAME')
        conv2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_bias))
        conv2 = tf.nn.lrn(conv2, depth_radius=5, bias=2, alpha=1e-4, beta=.75)
        layer3 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
        
        print("conv3", layer3.get_shape().as_list())
    # batch_size, 28, 28, 48
    # reshape
    layer3_shape = layer3.get_shape().as_list()
    nodes = layer3_shape[1] * layer3_shape[2] * layer3_shape[3]
    layer3_flatten = tf.reshape(layer3, [-1, nodes])
#     print(layer3_shape)
    
    with tf.variable_scope("layer4_fc1"):
        fc1_weights = get_weights([nodes, FC_SIZE1])
        fc1_bias = get_bias([FC_SIZE1])
        fc1 = tf.nn.relu(tf.matmul(layer3_flatten, fc1_weights) + fc1_bias)
        layer4 = tf.nn.dropout(fc1, dropout)
        
    with tf.variable_scope("layer5_fc2"):
        fc2_weights = get_weights([FC_SIZE1, FC_SIZE2])
        fc2_bias = get_bias([FC_SIZE2])
        fc2 = tf.nn.relu(tf.matmul(layer4, fc2_weights) + fc2_bias)
        layer5 = tf.nn.dropout(fc2, dropout)
    
    with tf.variable_scope("layer6_softmax"):
        softmax_weights = get_weights([FC_SIZE2, NUM_LABELS])
        softmax_bias = get_bias([NUM_LABELS, ])
        logits = tf.matmul(layer5, softmax_weights) + softmax_bias
        pred = tf.nn.softmax(logits)
    
    with tf.variable_scope("loss"):
        correct = tf.equal(tf.argmax(pred, 1), tf.argmax(labels, 1))
        acc = tf.reduce_mean(tf.cast(correct, tf.float32))
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
        
#         tf.summary.scalar('loss',loss)
#         tf.summary.scalar('acc',acc)
    return loss, acc, conv_res_weights


# In[ ]:





# In[ ]:


def trian(PRINT_LOSS_EVERY_ITE=500, PRINT_ACC_EVERY_ITE=1000):
    
    tf.reset_default_graph()
    
    train_dataset = dataset("tfrecords/MF_clf_train.tfrecords-*", 
                            batch_size=64, 
                            num_epochs=50, 
                            is_shuffle=False)
    test_dataset = dataset("tfrecords/MF_clf_test.tfrecords", 
                            batch_size=100, 
                            num_epochs=1)
    iterator = train_dataset.make_initializable_iterator()
    test_iterator = test_dataset.make_initializable_iterator()
    
    image_batch, label_batch = iterator.get_next()
    test_image_batch, test_label_batch = test_iterator.get_next()
    
    input_placeholder, labels_placeholder, dropout_placeholder = get_placeholder()
    loss, acc, weights= build_nn(input_placeholder, labels_placeholder, dropout_placeholder)
    
#     global_step = tf.Variable(0, trainable=False)
#     lr = tf.train.exponential_decay(1e-4, global_step, 1000, 0.95)
#     train = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss, global_step=global_step)
#     train = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9).minimize(loss)
    train = tf.train.AdamOptimizer(1e-4).minimize(loss)
    
    saver = tf.train.Saver()
    
    train_losses = []
    test_losses = []
    train_acces = []
    test_acces = []
    
    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        sess.run([iterator.initializer, test_iterator.initializer])
        
        #################### test data ####################
        dev_imgs, dev_labs = [], []
        while True:
            try:
                b_imgs, b_labs = sess.run([test_image_batch, test_label_batch])
                dev_imgs.append(b_imgs)
                dev_labs.append(b_labs)
            except tf.errors.OutOfRangeError:
                break
        dev_imgs = np.concatenate(dev_imgs)
        dev_labs = np.concatenate(dev_labs)
        ###################################################
        
        num_iterat = 1
        while True:
            try:
                _weights = sess.run(weights)
                weights.load(constraint_weights(_weights), sess)
                
                _image_batch, _label_batch = sess.run([image_batch, label_batch])
                _, _loss, _acc = sess.run([train, loss, acc], feed_dict={
                    input_placeholder: _image_batch,
                    labels_placeholder: _label_batch,
                    dropout_placeholder: 0.5 })
                num_iterat += 1
                
                train_acces.append(_acc)
                train_losses.append(_loss)
                
                if num_iterat > 20000:
                    break

                
                sys.stdout.write("\r ite {:>3} train loss:{:>6.2f}  train acc:{:.4f}".format(num_iterat, _loss, _acc))
                if num_iterat % PRINT_LOSS_EVERY_ITE == 0:
                    print("")
                    
    
                if num_iterat % PRINT_ACC_EVERY_ITE == 0:
                    _loss, _acc = sess.run([loss, acc], feed_dict={
                    input_placeholder : dev_imgs,
                    labels_placeholder : dev_labs,
                    dropout_placeholder : 1})

                    test_losses.append(_loss)
                    test_acces.append(_acc)
                    
                    print("\ntest loss = %.5f  test acc = %.6f" % (_loss, _acc) )
                    
                    if _acc > 0.99:
                        saver.save(sess, "save/MF_clf_model.ckpt")
                        break
            except tf.errors.OutOfRangeError:
                _weights = sess.run(weights)
                weights.load(constraint_weights(_weights), sess)
                print("training end")
                
                saver.save(sess, "save/MF_clf_model.ckpt")
                break
                
    
    plt.figure(figsize=(12, 8))
    plt.subplot(221)
    plt.plot(train_losses, c='b')
    plt.subplot(222)
    plt.plot(train_acces, c='b')
    plt.subplot(223)
    plt.plot(test_losses, c='r')
    plt.subplot(224)
    plt.plot(test_acces, c='r')
    plt.show()

# trian()


# In[ ]:





# In[ ]:





# # 加载训练好的模型进行结果验证

# In[ ]:


# 加载模型

input_placeholder, labels_placeholder, dropout_placeholder = get_placeholder()
loss, acc, weights= build_nn(input_placeholder, labels_placeholder, dropout_placeholder)

saver = tf.train.Saver()

sess = tf.Session()
saver.restore(sess, "save/MF_clf_model.ckpt")

_weights = sess.run(weights)
weights.load(constraint_weights(_weights), sess)
pred = tf.get_default_graph().get_tensor_by_name("layer6_softmax/Softmax:0")


# In[ ]:


def split64(gray_img):
    height, width = gray_img.shape
    # [num_raws, num_cols] are subimage numbers on vertical or horizontal direction
    num_raws, num_cols = height//64, width//64
    start_raws, start_cols = height%64//2, width%64//2
    sub_imgs = []
    indexes = [(i, j) for i in range(num_raws) for j in range(num_cols)]
    for i, j in indexes:
        x, y = start_cols + j * 64, start_raws + i * 64
        sub_img = gray_img[y:y+64, x:x+64]
        sub_imgs.append(sub_img)
    return np.array(sub_imgs)


# ## 在UCID小分辨率图片上测试效果

# In[ ]:


import pandas as pd

a_lst = []
for f in glob("dataset/ucid/*.tif"):
    try:
        g = cv2.imread(f)[:,:,1]
        imgs = split64(g)
        p_imgs = np.array([cv2.medianBlur(i, 5) for i in imgs])

        x = np.concatenate([imgs, p_imgs])
        x = x.reshape([-1, 64, 64, 1])
        x = x / 255
        y = np.zeros([x.shape[0]])
        y[x.shape[0]//2:] = 1
        
        y = pd.get_dummies(y).values

        _a = sess.run(acc, feed_dict={
            input_placeholder: x, 
            labels_placeholder: y, 
            dropout_placeholder: 1.0
        })
        a_lst.append(_a)
    except:
        continue
print("average accurracy on UCID is %.2f%%" % (np.array(a_lst).mean()*100) )


# In[ ]:


a_lst = []
for f in glob("dataset/ucid/*.tif"):
    try:
        g = cv2.imread(f)[:,:,1]
        imgs = split64(g)
        x = imgs.reshape([-1, 64, 64, 1])
        x = x / 255
        y = np.array([[1, 0] for _ in range(x.shape[0])])

        _a = sess.run(acc, feed_dict={
            input_placeholder: x, 
            labels_placeholder: y, 
            dropout_placeholder: 1.0
        })
        a_lst.append(_a)
    except:
        continue
print("average error on UCID is %.2f%%" % (100-np.array(a_lst).mean()*100) )


# __This means an image is recongnize as altered one if it has 7.55% or more blocks(64 by 64) diagnosised as positive , in which 7.55% is a threshold computed on UCID dataset.__  

# ## 人物照片上测试分类准确率

# In[ ]:


g = cv2.imread("./yy.jpg")[:,:,1]
imgs = split64(g)
p_imgs = np.array([cv2.medianBlur(i, 5) for i in imgs])

x = np.concatenate([imgs, p_imgs])
x = x.reshape([-1, 64, 64, 1])
x = x / 255
y = np.zeros([x.shape[0]])
y[x.shape[0]//2:] = 1
import pandas as pd
y = pd.get_dummies(y).values

a = sess.run(acc, feed_dict={
    input_placeholder: x, 
    labels_placeholder: y, 
    dropout_placeholder: 1.0
})
print("acc %.2f%% over %d blocks" % (a*100, x.shape[0]))


# ## 检测修过图的照片

# In[ ]:


gray_img = cv2.imread("me.jpg", cv2.IMREAD_ANYCOLOR)[:,:,1]

height, width = gray_img.shape
# [num_raws, num_cols] are subimage numbers on vertical or horizontal direction
num_raws, num_cols = height//64, width//64
start_raws, start_cols = height%64//2, width%64//2
img_boxes = np.copy(gray_img)  # for test perpose

indexes = [(i, j) for i in range(num_raws) for j in range(num_cols)]

for i, j in indexes:
    x, y = start_cols + j * 64, start_raws + i * 64
    sub_img = gray_img[y:y+64, x:x+64]
    p = sess.run(pred, feed_dict={input_placeholder: sub_img.reshape([1,64,64,1])/255, 
                                    dropout_placeholder:1.0})
    if np.argmax(p, 1) == 1:
        cv2.rectangle(img_boxes, (x, y), (x+64, y+64), (0, 255, 0), 5)
    
plt.figure(figsize=(9, 16))
plt.imshow(img_boxes, cmap="gray");


# ## 原图 vs PS后图片

# In[ ]:


# 检测函数
def detect(file):
    gray_img = cv2.imread(file, cv2.IMREAD_ANYCOLOR)[:,:,1]
    height, width = gray_img.shape
    # [num_raws, num_cols] are subimage numbers on vertical or horizontal direction
    num_raws, num_cols = height//64, width//64
    start_raws, start_cols = height%64//2, width%64//2
    sub_imgs = []
    img_boxes = np.copy(gray_img)  # for test perpose
    indexes = [(i, j) for i in range(num_raws) for j in range(num_cols)]
    for i, j in indexes:
        x, y = start_cols + j * 64, start_raws + i * 64
        sub_img = gray_img[y:y+64, x:x+64]
        sub_imgs.append(sub_img)
        p = sess.run(pred, feed_dict={input_placeholder: sub_img.reshape([1,64,64,1])/255, 
                                        dropout_placeholder:1.0})
        if np.argmax(p, 1) == 1:
            cv2.rectangle(img_boxes, (x, y), (x+64, y+64), (0, 255, 0), 5)
    return img_boxes


# In[ ]:


r1 = detect('tt.jpg')
r2 = detect('tt-ps.jpg')

plt.figure(figsize=(18, 18))
plt.subplot(211)
plt.imshow(r1, cmap='gray')
plt.title('unchange img')
plt.subplot(212)
plt.imshow(r2, cmap='gray')
plt.title('after PS')
plt.show()


# In[ ]:





# ## residual层的特征可视化

# In[ ]:


res = tf.get_default_graph().get_tensor_by_name('layer1_conv_res/BiasAdd:0')


# In[ ]:


f = "dataset/ucid/ucid00004.tif"

g = cv2.imread(f)[:,:,1]
imgs = split64(g)
x = np.array([cv2.medianBlur(i, 5) for i in imgs])
x = x.reshape([-1, 64, 64, 1])
x = x / 255

res_o = sess.run(res, feed_dict={input_placeholder: x})
res_imgs = [[x[i,:,:,0]]+[res_o[i, :,:, j] for j in range(12)] for i in range(5)]

fig, axes = plt.subplots(nrows=5, ncols=12, sharex=True, sharey=True, figsize=(12*1.5,5*1.5))

for images, row in zip(res_imgs, axes):
    for img, ax in zip(images, row):
        ax.imshow(img, cmap='Greys_r')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

fig.tight_layout(pad=0.1)
print("residual output among several changed images, x = channels, y = inputs")
plt.show()

print("residual output among several original images")
g = cv2.imread(f)[:,:,1]
imgs = split64(g)
x = imgs.reshape([-1, 64, 64, 1])
x = x / 255

res_o = sess.run(res, feed_dict={input_placeholder: x})
res_imgs = [[x[i,:,:,0]]+[res_o[i, :,:, j] for j in range(12)] for i in range(5)]

fig, axes = plt.subplots(nrows=5, ncols=12, sharex=True, sharey=True, figsize=(12*1.5,5*1.5))

for images, row in zip(res_imgs, axes):
    for img, ax in zip(images, row):
        ax.imshow(img, cmap='Greys_r')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

fig.tight_layout(pad=0.1)
plt.show()


# In[ ]:





# ## test on a small PS image dataset find in Baidu Image

# In[ ]:


results = []
for f in glob("test_img/*.*"):
#     try:
    g = cv2.imread(f)[:,:,1]
    imgs = split64(g)
    x = imgs.reshape([-1, 64, 64, 1])
    x = x / 255

    _p = sess.run(pred, feed_dict={
        input_placeholder: x, 
        dropout_placeholder: 1.0
    })
    score = np.argmax(_p, 1).mean()
    results.append(score)
#     if score > 0.01:
    r1 = detect(f)
    plt.imshow(r1, cmap='gray')
    plt.show()
#     except:
#         continue
results = np.array(results)


# In[ ]:





# In[ ]:


[n.name for n in tf.get_default_graph().as_graph_def().node]

