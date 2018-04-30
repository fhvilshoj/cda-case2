import os
import os.path as path
import shutil
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

def _move_to_data_folder(config, files):
    
    i, j = 0, 0
    def _output_path(label):
        prefix = "foggy" if label else "clear"
        k = i if label else j
        return os.path.join(config.destination, f'{prefix}-{k:04}.jpg')
    
    for f, l in files:
        shutil.copyfile(f, _output_path(l))
        if l:
            i = i + 1
        else:
            j = j + 1

    
def main(config):
    all_jpgs = find_all_jpgs()
    random.shuffle(all_jpgs)

    # Filter out clear and foggy
    # 0 is clear, 1 is foggy
    clear_jpgs = []
    foggy_jpgs = []
    for jpg in all_jpgs:
        if jpg[1]:
            foggy_jpgs.append(jpg)
        else:
            clear_jpgs.append(jpg)
                
    foggy_jpgs.extend(clear_jpgs[:len(foggy_jpgs)])
    all_jpgs = foggy_jpgs
    random.shuffle(all_jpgs)
    print(f"Images in total: {len(all_jpgs)}")
        
    _move_to_data_folder(config, all_jpgs)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='generate_tfrecords.py',
                                     description='''Locates images in ../data/valid and transforms
                                     them into TFRecords.
                                     ''',
                                     epilog='Good luck ;-)')
    parser.add_argument('-d', '--destination', default='../data/dataset',
                        help='Destination directory for selected files')
    
    args = parser.parse_args()
        
    main(args)
