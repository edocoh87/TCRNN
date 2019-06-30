import numpy as np

class MNISTImageGenerator(object):
    """ Generate sequence of data with dynamic length.
    This class generate samples for training:
    - Class 0: linear sequences (i.e. [0, 1, 2, 3,...])
    - Class 1: random sequences (i.e. [1, 3, 10, 7,...])
    NOTICE:
    We have to pad each sequence to reach 'max_seq_len' for TensorFlow
    consistency (we cannot feed a numpy array with inconsistent
    dimensions). The dynamic calculation will then be perform thanks to
    'seqlen' attribute that records every actual sequence length.
    """
    def __init__(self, n_samples=1000, max_seq_len=10, min_seq_len=1, mode='sum',
        train=True, path="/Users/edock/Desktop/Desktop - Edo’s MacBook Pro/Work/invariant_architecture/DeepSets/DigitSum/data/"):
        def _load_mnist(pack=0):
            # path = 
            # # path = "/specific/netapp5_2/gamir/edocohen/DeepSets/DigitSum/data/"
            img = np.load(path + "mnist8m_{}_features.npy".format(pack))
            label = np.load(path + "mnist8m_{}_labels.npy".format(pack))
            return img, label

        if train:
            pack = 0
        else:
            pack = np.random.randint(1,8)

        img, labels = _load_mnist(pack)
        mask = labels>0
        _img = img[mask]
        _labels = labels[mask]
        # filter out zeros
        self.data = []
        self.labels = []
        self.seqlen = []
        c = 0
        for i in range(n_samples):
            # Random sequence length
            _seqlen = np.random.randint(min_seq_len, max_seq_len)
            # Monitor sequence length for TensorFlow dynamic calculation
            self.seqlen.append(_seqlen)
            curr_idxs = np.random.choice(len(_img), _seqlen)
            s_imgs = _img[curr_idxs,:]
            # pad the sequence with zeros.
            s = np.concatenate([s_imgs, np.zeros((max_seq_len-_seqlen, s_imgs.shape[1]))])
            self.data.append(s)

            if mode == 'sum':
                label = sum(np.array(_labels[curr_idxs]))
            elif mode == 'max':
                label = np.max(np.array(_labels[curr_idxs]))
            elif mode == 'parity':
                label = sum(np.array(_labels[curr_idxs]))%2

            self.labels.append([label])
            # self.labels.append([sum(np.array(_labels[curr_idxs]))])
        self.batch_id = 0


    def next(self, batch_size=np.inf):
        """ Return a batch of data. When dataset end is reached, start over.
        """
        if self.batch_id == len(self.data):
            self.batch_id = 0
        batch_data = (self.data[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        batch_labels = (self.labels[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        batch_seqlen = (self.seqlen[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        self.batch_id = min(self.batch_id + batch_size, len(self.data))
        return batch_data, batch_labels, batch_seqlen


if __name__=='__main__':
    sample = MNISTImageGenerator()
    for i in range(5):
        batch_data, batch_labels, batch_seqlen = sample.next(10)
        print(batch_data[0].shape)
        print(batch_labels)
        print(batch_seqlen)