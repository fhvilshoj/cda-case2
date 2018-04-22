import os
import os.path as path
import argparse
import random

import numpy as np
import tensorflow as tf
import skimage.io as io

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
    

def write_images_to_tfrecord(image_paths, dest):
    writer = tf.python_io.TFRecordWriter(dest)
    
    for (image_path, label) in image_paths:
        img = io.imread(image_path)
        height, width, depth = img.shape
        
        img_raw = tf.compat.as_bytes(img.tostring())
        
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(height),
            'width': _int64_feature(width),
            'label': _int64_feature(label),
            'image_raw': _bytes_feature(img_raw)
        }))
        writer.write(example.SerializeToString())
    writer.close()
    print(f"Wrote {len(image_paths)} images to {dest}")

def main(config):
    all_jpgs = find_all_jpgs()
    random.shuffle(all_jpgs)

    if config.max_samples > 0:
        all_jpgs = all_jpgs[:config.max_samples]
    
    full_records = len(all_jpgs) // config.samples_per_file
    for i in range(full_records):
        write_images_to_tfrecord(all_jpgs[i*config.samples_per_file:(i+1)*config.samples_per_file],
                                 os.path.join(config.destination, f'{config.prefix}-{i+1}.tfrecords'))

    write_images_to_tfrecord(all_jpgs[full_records*config.samples_per_file:],
                             os.path.join(config.destination, f'{config.prefix}-{full_records + 1}.tfrecords'))


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
    parser.add_argument('--samples-per-file', default=15000, type=int,
                        help='Number of samples to store in a file')
    args = parser.parse_args()
        
    main(args)
