import os
import os.path as path

labels = ['./clear', './foggy']

all_jpgs = []
total_file_count = 0
total_duplicates = 0

def merge_lists(l1, l2):
    i1 = i2 = 0
    duplicates = []
    merge = []
    
    while i1 < len(l1) and i2 < len(l2):
        if l1[i1] == l2[i2]:
            duplicates.append(l2[i2])
            i1 = i1 + 1
            i2 = i2 + 1
        elif l1[i1] < l2[i2]:
            merge.append(l1[i1])
            i1 = i1 + 1
        else:
            merge.append(l2[i2])
            i2 = i2 + 1
    while i1 < len(l1):
        merge.append(l1[i1])
        i1 = i1 + 1
    while i2 < len(l2):
        merge.append(l2[i2])
        i2 = i2 + 1

    return merge, duplicates

for label in labels:
    for subdir in os.listdir(label):
        files = [f for f in os.listdir(os.path.join(label, subdir)) if f.endswith(".jpg")]
        total_file_count = total_file_count + len(files)
        files.sort()
        
        if not all_jpgs:
            all_jpgs = files
        else:
            all_jpgs, duplicates = merge_lists(all_jpgs, files)
            total_duplicates = total_duplicates + len(duplicates)
            
            if duplicates:
                print(f"Duplicates: {len(duplicates)}")

print(f"Duplicates in total: {total_duplicates}")
    
