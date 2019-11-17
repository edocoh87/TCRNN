import numpy as np

from DataGenerator import DataGenerator

def to_onehot(arr, vec_len):
    res = np.zeros((len(arr), vec_len))
    res[np.arange(len(arr)), arr] = 1.0
    return res

class DigitSequenceGenerator(DataGenerator):
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
    def __init__(self, n_samples=1000, max_seq_len=20, min_seq_len=3,
                 max_value=9, mode='sum', **kwargs):
        self.data = []
        self.labels = []
        self.seqlen = []
        for i in range(n_samples):
            # Random sequence length
            _seqlen = np.random.randint(min_seq_len, max_seq_len)
            # Monitor sequence length for TensorFlow dynamic calculation
            self.seqlen.append(_seqlen)
            # Add a random or linear int sequence (50% prob)
            # if random.random() < .5:
            #     # Generate a linear sequence
            #     rand_start = random.randint(0, max_value - len)
            #     s = [[float(i)/max_value] for i in
            #          range(rand_start, rand_start + len)]
            #     # Pad sequence for dimension consistency
            #     s += [[0.] for i in range(max_seq_len - len)]
            #     self.data.append(s)
            #     self.labels.append([1., 0.])
            # else:
            # Generate a random sequence
            # s = [[float(random.randint(0, max_value))/max_value]
            s = [[float(np.random.randint(0, max_value))]
                 for i in range(_seqlen)]
            # Pad sequence for dimension consistency
            s += [[0.] for i in range(max_seq_len - _seqlen)]
            self.data.append(s)
            # print(sum(np.array(s)))
            # print(sum(np.array(s)).shape)
            # print([np.max(np.array(s))])
            # print(np.max(np.array(s)).shape)
            # exit()
            # self.labels.append([np.max(np.array(s))])
            if mode == 'sum':
                label = sum(np.array(s))
            elif mode == 'max':
                label = [np.max(np.array(s))]
            elif mode == 'prty':
                #label = to_onehot(sum(np.array(s))%2, 2)
                label = int((sum(np.array(s))%2)[0])
            elif mode == 'anmly':
                # label is 1 if there are three consecutive identical numbers
                # or if the sum of the sequence is greater than 120.
                label = 0 
                # print(np.split(s, np.where(np.diff(s) != 0)[0]+1))
                _s = np.array(s).ravel()
                # print(_s)
                # print(np.split(_s, np.where(np.diff(_s) != 0)[0]+1))
                for _ in np.split(_s, np.where(np.diff(_s) != 0)[0]+1):
                    # print(_)
                    if len(_) > 2 and _[0] > 0:
                        # print(_)
                        # if we're here, there are consecutive numbers in the array.
                        label = 1
                if np.sum(_s) > 150:
                    label = 1
                # label = to_onehot(label, 2)

            self.labels.append(label)
        # print(self.labels)
            # self.labels.append(sum(np.array(s)))
        if mode in ['prty', 'anmly']:
            self.labels = to_onehot(self.labels, 2)
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
    sample = DigitSequenceGenerator(n_samples=20, mode='sum', min_seq_len=3, max_seq_len=10)
    # exit()
    for i in range(1):
        batch_data, batch_labels, batch_seqlen = sample.next(1)
        print(batch_data)
        print(batch_seqlen)
        print(batch_labels)
        # print(np.array(batch_data))
        # print(batch_labels)
        # print(batch_seqlen)
