import numpy as np
from tqdm import tqdm
import h5py

from DataGenerator import DataGenerator

def rotate_z(theta, x):
    theta = np.expand_dims(theta, 1)
    outz = np.expand_dims(x[:,:,2], 2)
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    xx = np.expand_dims(x[:,:,0], 2)
    yy = np.expand_dims(x[:,:,1], 2)
    outx = cos_t * xx - sin_t*yy
    outy = sin_t * xx + cos_t*yy
    return np.concatenate([outx, outy, outz], axis=2)
    
def augment(x):
    bs = x.shape[0]
    #rotation
    thetas = np.random.uniform(-0.1, 0.1, [bs,1])*np.pi
    rotated = rotate_z(thetas, x)
    #scaling
    scale = np.random.rand(bs,1,3)*0.45 + 0.8
    return rotated*scale

def standardize(x):
    clipper = np.mean(np.abs(x), (1,2), keepdims=True)
    z = np.clip(x, -100*clipper, 100*clipper)
    mean = np.mean(z, (1,2), keepdims=True)
    std = np.std(z, (1,2), keepdims=True)
    return (z-mean)/std

def to_onehot(arr, vec_len):
    res = np.zeros((len(arr), vec_len))
    res[np.arange(len(arr)), arr] = 1.0
    return res

class PointCloudGenerator(DataGenerator):
    # def __init__(self, fname='../DeepSets/PointClouds/ModelNet40_cloud_from_edo.h5', down_sample=100, do_standardize=True, do_augmentation=True, train=True):
    def __init__(self, fname='../PointClouds/ModelNet40_cloud.h5', down_sample=100, do_standardize=True, do_augmentation=True, train=True):

        self.fname = fname
        self.down_sample = down_sample

        if train:
            prefix = 'tr_'
            make_val_data = True
        else:
            prefix = 'test_'
            make_val_data = False
            do_augmentation = False

        with h5py.File(fname, 'r') as f:
            self.data = np.array(f[prefix + 'cloud'])
            self.labels = np.array(f[prefix + 'labels'])

        self.val_data = None
        self.val_labels = None
        if make_val_data:
            # shuffle the data to avoid getting the same validation set each time.
            rng_state = np.random.get_state()
            np.random.shuffle(self.data)
            np.random.set_state(rng_state)
            np.random.shuffle(self.labels)
            self.val_data = self.data[-500:]
            self.val_labels = self.labels[-500:]
            self.data = self.data[:-500]
            self.labels = self.labels[:-500]

        self.n_classes = np.max(self.labels) + 1

            # self._train_data = np.array(f['tr_cloud'])
            # self._train_label = np.array(f['tr_labels'])
            # self._test_data = np.array(f['test_cloud'])
            # self._test_label = np.array(f['test_labels'])
        
        # self.num_classes = np.max(self._train_label) + 1

        self.prep1 = standardize if do_standardize else lambda x: x
        self.prep2 = (lambda x: augment(self.prep1(x))) if do_augmentation else self.prep1

        # select the subset of points to use throughout beforehand
        self.perm = np.random.permutation(self.data.shape[1])[::self.down_sample]
        self.batch_id = 0
    
    def get_validation(self):
        batch_data = (self.val_data[:, self.perm])
        batch_labels = to_onehot(self.val_labels, self.n_classes)
        return self.prep2(batch_data), batch_labels

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
            
        end_idx = min(self.batch_id + batch_size, len(self.data))
        batch_data = (self.data[self.batch_id:end_idx, self.perm])
        batch_labels = to_onehot(self.labels[self.batch_id:end_idx], self.n_classes)
        self.batch_id = min(self.batch_id + batch_size, len(self.data))
        return self.prep2(batch_data), batch_labels


if __name__=='__main__':
    sample = PointCloudGenerator('../PointClouds/ModelNet40_cloud.h5', down_sample=1000)
    # sample_test = PointCloudGenerator('../PointClouds/ModelNet40_cloud.h5', mode='test')
    # print(sample.next(2)[1].shape)
    print(sample.data.shape)
    # for i in range(5):
    #     batch_data, batch_labels = sample.next(10)
        # print(batch_data.shape)
        # print(batch_labels.shape)
