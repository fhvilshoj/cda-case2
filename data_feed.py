import os
import os.path
import numpy as np
import random
import skimage.io as io
import skimage.color as color

import pickle

train_file = '../data/dataset/data_train.pkl'
test_file = '../data/dataset/data_test.pkl'

class Reader:
    def __init__(self):
        random.seed(1337)

        if not os.path.exists(train_file) or not os.path.exists(test_file):
            print(f"Couldn't find `{train_file}` or `{test_file}`")
            print("Generating them..")
            data = self._read_and_store_data()
        
        self.train_images = []
        self.tran_labels = []
        self.test_images = []
        self.test_labels = []

    def _read_and_store_data(self):
        base_dir = '../data/dataset'
        paths = [os.path.join(base_dir, f) for f in os.listdir(base_dir) if f.endswith('.jpg')]
        random.shuffle(paths)

        train = int(len(paths) * 0.6)
        train_files = paths[:train]
        test_files = paths[train:]
        first = True
        for file_set, dest in [(train_files, train_file), (test_files, test_file)]:
            images = []
            labels = []

            for path in file_set:
                img = io.imread(path)[:170]
                gray = color.rgb2gray(img)
                
                if first:
                    print(gray[100, 100:110])
                    first = False

                img = ((gray - .5) * 2).astype(np.float32)
            
                label = 1 if "foggy" in path else 0
                images.append(img)
                labels.append(label)
        
            images = np.array(images)
            labels = np.array(labels)

            data = {
                'images': images,
                'labels': labels
            }

            with open(dest, 'wb') as f:
                pickle.dump(data, f)

    def data(self, train=True):
        if train and not os.path.exists(train_file):
            raise FileNotFoundError()
        elif not os.path.exists(test_file):
            raise FileNotFoundError()
        
        with open(train_file if train else test_file, 'rb') as f:
            data = pickle.load(f)

        return data['images'], data['labels']




if __name__ == '__main__':
    main()
