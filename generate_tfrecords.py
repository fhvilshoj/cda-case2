import os
import os.path as path
import argparse
import random

import numpy as np
import tensorflow as tf
import skimage.io as io

from color_converter import rgb_to_hsv

#             0           1
labels = ['../data/valid/clear', '../data/valid/foggy']

def find_all_jpgs():
    all_jpgs = []

    for label, directory in enumerate(labels):
        for subdir in os.listdir(directory):
            files = [(os.path.join(directory, subdir, f), label)
                     for f in os.listdir(os.path.join(directory, subdir))
                     if f.endswith(".jpg")]
            if not all_jpgs:
                all_jpgs = files
            else:
                all_jpgs.extend(files)
    return all_jpgs

    
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    

def _add_inverse_channel(img):
    inverse = np.concatenate([img, 1-img], axis=2)
    shape = inverse.shape

    inverse = inverse.reshape((shape[0], shape[1], 2, 3))
    inverse = np.transpose(inverse, [0, 1, 3, 2])

    inverse = np.max(inverse, axis=3)
    inverse = (inverse - 0.5) * 2

    hsv_img = rgb_to_hsv(img)
    hsv_inverse = rgb_to_hsv(inverse)

    difference = np.abs(hsv_img[:,:,0] - hsv_inverse[:,:,0]).reshape((shape[0], shape[1], 1))
    diff_uint8 = (difference * 255).astype(np.uint8)

    return np.concatenate([img, diff_uint8], axis=2)

    
def write_images_to_tfrecord(image_paths, dest, config):
    writer = tf.python_io.TFRecordWriter(dest)
    
    for (image_path, label) in image_paths:
        img = io.imread(image_path)

        if config.add_inverse_channel:
            img = _add_inverse_channel(img)
        
        img_raw = tf.compat.as_bytes(img.tostring())        
        example = tf.train.Example(features=tf.train.Features(feature={
            'label': _int64_feature(label),
            'image_raw': _bytes_feature(img_raw)
        }))
        writer.write(example.SerializeToString())
    writer.close()
    print(f"Wrote {len(image_paths)} images to {dest}")

def _output_path(config, i):
    addition = "_d4" if config.add_inverse_channel else "_d3"
    return os.path.join(config.destination, f'{config.prefix}-{i+1}{addition}.tfrecords')
    
def main(config):
    all_jpgs = find_all_jpgs()
    random.shuffle(all_jpgs)

    if config.max_samples > 0:
        all_jpgs = all_jpgs[:config.max_samples]
    elif config.split_equal:
        # Filter out clear and foggy
        # 0 is clear, 1 is foggy
        clear_jpgs = []
        foggy_jpgs = []
        for jpg in all_jpgs:
            if jpg[1]:
                foggy_jpgs.append(jpg)
            else:
                clear_jpgs.append(jpg)
        print(len(foggy_jpgs),
              len(clear_jpgs))
              
        foggy_jpgs.extend(clear_jpgs[:len(foggy_jpgs)])
        all_jpgs = foggy_jpgs
        random.shuffle(all_jpgs)
        print(f"Images in total: {len(all_jpgs)}")
    
    full_records = len(all_jpgs) // config.samples_per_file
    for i in range(full_records):
        write_images_to_tfrecord(all_jpgs[i*config.samples_per_file:(i+1)*config.samples_per_file],
                                 _output_path(config, i),
                                 config)

    write_images_to_tfrecord(all_jpgs[full_records*config.samples_per_file:],
                             _output_path(config, full_records),
                             config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='generate_tfrecords.py',
                                     description='''Locates images in ../data/valid and transforms
                                     them into TFRecords.
                                     ''',
                                     epilog='Good luck ;-)')
    parser.add_argument('-d', '--destination', default='../data/tfrecords',
                        help='Destination directory for TFRecord files')
    parser.add_argument('-p', '--prefix', default='shuffle',
                        help='Prefix for the produced files')
    parser.add_argument('--max-samples', default=-1, type=int,
                        help='Max number of samples to convert to TFRecords')
    parser.add_argument('--split-equal', action='store_true',
                        help='Use all foggy images and sample as many clear')
    parser.add_argument('--samples-per-file', default=15000, type=int,
                        help='Number of samples to store in a file')
    parser.add_argument('-ai', '--add-inverse-channel', action='store_true',
                        help='Add the semi-inverse of the image to the data')
    
    args = parser.parse_args()
        
    main(args)
