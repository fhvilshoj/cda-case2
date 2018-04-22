import os
import os.path
from subprocess import run
from subprocess import PIPE
import sys
import shutil

import cv2
import numpy as np

dirs = [f'data/clear/bll{i}' for i in range(1,11)]
dirs.append('data/foggy/bll')

dest = 'data/valid/'
duplicate_dest = 'data/duplicates/'

def check_files_match(f1, f2):
    a = cv2.imread(f1)
    b = cv2.imread(f2)
    difference = cv2.subtract(a, b)    
    return not np.any(difference)
    
broken_cnt = 0
copy_cnt = 0
img_shapes = {}

remove_queue = []

for d in dirs:
    print('# ' * 10, d ,' #'*10)
    path = d.split('/')
    ddest = os.path.join(dest, path[1], path[2])
    dupdest = os.path.join(duplicate_dest, path[1], path[2])
    
    if not os.path.exists(ddest):
        os.makedirs(ddest)
    if not os.path.exists(dupdest):
        os.makedirs(dupdest)

    imgs = os.listdir(d)
    imgs.sort()

    last_shape = ''
    last_size = 0
    last_file = ''
    
    for img in imgs:
        img_pth = os.path.join(d, img)
        
        res = run(['identify', img_pth], stdout=PIPE, stderr=PIPE)
        
        if res.stderr:
            cnt = broken_cnt + 1
        else:
            img_info = res.stdout.decode('utf-8').split()
            shape = img_info[2]
            file_size = img_info[6][:-2]

            img_shapes[shape] = 1
            
            if file_size == last_size and last_shape == shape:
                if check_files_match(img_pth, os.path.join(d, last_file)):
                    copy_cnt = copy_cnt + 1
                    os.rename(img_pth, os.path.join(dupdest, img))
                    continue

            # Crop
            res = run(['convert', img_pth, '-crop', '384x272+0+16', os.path.join(ddest, img)])
            remove_queue.append(img_pth)
            
            last_file = img
            last_size = file_size
            last_shape = shape

        if len(remove_queue) > 2000:
            for f in remove_queue[:1000]:
                os.remove(f)
            remove_queue = remove_queue[1000:]

# Remove remaining files
for f in remove_queue:
    os.remove(f)


print(img_shapes)
print(f'Broken images: {broken_cnt}, Copy images: {copy_cnt}')
