#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
np.random.seed(2018)
import cv2
import tensorflow as tf
from tqdm import tqdm
from glob import glob
import os


# In[ ]:


def split_256(gray_img):
    height, width = gray_img.shape
    num_raws, num_cols = height//256, width//256
    start_raws, start_cols = height%256//2, width%256//2
    
    sub_imgs = []
        
    indexes = [(i, j) for i in range(num_raws) for j in range(num_cols)]
    
    for i, j in indexes:
        x, y = start_cols + j * 256, start_raws + i * 256
        sub_img = gray_img[y:y+256, x:x+256]
        sub_imgs.append(sub_img)
    return np.array(sub_imgs)


def images_square_grid(images):
    """
    Save images as a square grid
    :param images: Images to be used for the grid
    :param mode: The mode to use for images
    :return: Image of images in a square grid
    """
    # Get maximum size for square grid of images
    save_size = math.floor(np.sqrt(images.shape[0]))

    images = (((images - images.min()) * 255) / (images.max() - images.min())).astype(np.uint8)

    images_in_square = np.reshape(
            images[:save_size*save_size],
            (save_size, save_size, images.shape[1], images.shape[2]))

    # Combine images to grid image
    new_im = np.ones((save_size*256, save_size*256), dtype=np.uint8)
    for col_i, col_images in enumerate(images_in_square):
        for image_i, image in enumerate(col_images):
            new_im[col_i * 256: col_i * 256 + 256, 
                   image_i * 256: image_i * 256 + 256] = image

    return new_im


# In[ ]:


img_file_lst = glob("dataset/ddimgdb/*")
np.random.shuffle(img_file_lst)
print("    %d items" % len(img_file_lst))
print("    show top 5 items")
print("\n".join(img_file_lst[:5]))

output_path= "dataset/db256_1"
if not os.path.exists(output_path):
    os.mkdir(output_path)

all_files = len(img_file_lst)

for i_file, file in tqdm(enumerate(img_file_lst), total=all_files):
    name = file.split("\\")[1].split(".")[0]
                       
    img = cv2.imread(file, cv2.IMREAD_COLOR)
    gray_img = cv2.split(img)[1]   # r,g,b split
    
    sub_imgs = split_256(gray_img)
    nums_imgs = len(sub_imgs)
    
    for i, img in enumerate(sub_imgs):
        cv2.imwrite("%s/%s-%.3d-of-%.3d.jpg" % (output_path, name, i+1, nums_imgs), img)


# In[ ]:





# In[ ]:


def __int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def __bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# In[ ]:


files = glob(output_path)


# In[ ]:


n_file = files[:50000]
p_file = ["TAG:positive" + each for each in n_file]


# In[ ]:


train_file = n_file + p_file
np.random.shuffle(train_file)

test_file = files[-800:] + ["TAG:positive"+each for each in files[-800:]]
np.random.shuffle(test_file)


# In[ ]:


print("train size:", len(train_file), "  test size:", len(test_file))


# In[ ]:


instances_per_shard = 5000
num_shards = len(train_file) // instances_per_shard

for i in range(num_shards):
    
    tfrecord = "tfrecords/MF_clf_train.tfrecords-%.2d-of-%.2d" % (i+1, num_shards)
    writer = tf.python_io.TFRecordWriter(tfrecord)
    
    s_i = instances_per_shard * i
    e_i = s_i + instances_per_shard
    for file in tqdm( train_file[s_i : e_i] ):
        if file.startswith("TAG:positive"):
            img = cv2.imread(file[12:], cv2.IMREAD_GRAYSCALE)
            img = cv2.medianBlur(img, ksize=5)

            label = 1
        else:
            img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            label = 0
        
        image_raw = img.reshape([256 * 256])
        image_raw = image_raw.tostring()

        example = tf.train.Example(features=tf.train.Features(feature={
            'image_raw':__bytes_feature(image_raw),
            'label':__int64_feature(label),
            'path':__bytes_feature( bytes(file, encoding='utf-8') ) 
        }))
        writer.write(example.SerializeToString())
    writer.close()


# In[ ]:


instances_per_shard = 5000
num_shards = len(test_file) // instances_per_shard

tfrecord = "tfrecords/MF_clf_test.tfrecords"
writer = tf.python_io.TFRecordWriter(tfrecord)

for file in tqdm( test_file ):
    if file.startswith("TAG:positive"):
        img = cv2.imread(file[12:], cv2.IMREAD_GRAYSCALE)
        img = cv2.medianBlur(img, ksize=5)

        label = 1
    else:
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        label = 0

    image_raw = img.reshape([256 * 256])
    image_raw = image_raw.tostring()

    example = tf.train.Example(features=tf.train.Features(feature={
        'image_raw':__bytes_feature(image_raw),
        'label':__int64_feature(label),
        'path':__bytes_feature( bytes(file, encoding='utf-8') ) 
    }))
    writer.write(example.SerializeToString())
writer.close()

