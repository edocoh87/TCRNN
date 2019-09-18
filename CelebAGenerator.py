import os
import pandas as pd
import numpy as np
from PIL import Image

from DataGenerator import DataGenerator

def load_row(images, base_dir):
    curr_img = '{:06}.jpg'
    return [np.asarray(Image.open(os.path.join(base_dir, curr_img.format(img)))) for img in images]


def load_dataset_from_csv(fname, img_base_dir):
    df = pd.read_csv(fname)
    # convert pandas strings back to numpy arrays
    df['pos'] = df.apply(lambda x: np.fromstring(x['pos'][1:-1], dtype=int, sep=' '), axis=1)
    df['neg'] = df.apply(lambda x: np.fromstring(x['neg'][1:-1], dtype=int, sep=' '), axis=1)
    images = []
    # get the images corresponding to the indices.
    # the negative example is always the last.
    for i, row in df.iterrows():
        images.append(np.array(
            load_row(row['pos'], base_dir=img_base_dir) + 
            load_row(row['neg'], base_dir=img_base_dir)
        ))
    return images

def standardize(x):
    new_x = []
    for _x in x:
        _x = _x.astype(np.float32)
        _x = _x / 255.0
        new_x.append(_x)
    return new_x

def flatten_images(x):
    new_x = []
    for _x in x:
        new_x.append(_x.reshape(7, -1))
    return new_x

def to_onehot(arr, vec_len):
    res = np.zeros((len(arr), vec_len))
    res[np.arange(len(arr)), arr] = 1.0
    return res

class CelebAGenerator(DataGenerator):
    # def __init__(self, fname='../DeepSets/PointClouds/ModelNet40_cloud_from_edo.h5', down_sample=100, do_standardize=True, do_augmentation=True, train=True):
    def __init__(self, base_dir='../CelebA', do_standardize=True, train=True):

        if train:
            fname = 'dataset_1_train.csv'
        else:
            fname = 'dataset_1_test.csv'
        
        self.data = load_dataset_from_csv(
            os.path.join(base_dir, fname),
            os.path.join(base_dir, 'img_align_celeba'))
        np.random.shuffle(self.data)
        # initial labels is all set to last image.
        self.labels = np.array([len(self.data[0])-1] * len(self.data))
        
        self.shuffle_all_datasets()
        self.n_classes = len(self.data[0])
        self.batch_id = 0

        self.prep1 = standardize if do_standardize else lambda x: x
        self.prep2 = lambda x: flatten_images(self.prep1(x))

    # iterate over rows and in each row shuffle it's internal order and update the label accordingly.
    def shuffle_all_datasets(self):
        def shuffle_dataset(images, label):
            perm = np.random.permutation(len(images))
            images = images[perm]
            label = np.argwhere(perm==label)[0][0]
            return images, label

        for i in range(len(self.data)):
            self.data[i], self.labels[i] = shuffle_dataset(self.data[i], self.labels[i])
        
    def next(self, batch_size=np.inf):
        """ Return a batch of data. When dataset end is reached, start over.
        """
        if self.batch_id == len(self.data):
            self.batch_id = 0
            # shuffle the data each pass over it.
            rng_state = np.random.get_state()
            np.random.shuffle(self.data)
            np.random.set_state(rng_state)
            np.random.shuffle(self.labels)
            self.shuffle_all_datasets()
            
        end_idx = min(self.batch_id + batch_size, len(self.data))
        batch_data = self.data[self.batch_id:end_idx]

        batch_labels = to_onehot(self.labels[self.batch_id:end_idx], self.n_classes)
        self.batch_id = min(self.batch_id + batch_size, len(self.data))
        return self.prep2(batch_data), batch_labels


if __name__=='__main__':
    sample = CelebAGenerator()
    # print(sample.data[0].shape)
    img = sample.next(2)
    print(img[0][0])
    print(img[0][0].shape)
    print(img[0][1].shape)

    # print(sample.next(1))
    # print(sample.data.shape)
    # for i in range(5):
    #     batch_data, batch_labels = sample.next(10)
        # print(batch_data.shape)
        # print(batch_labels.shape)
