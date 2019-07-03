import numpy as np

class DigitSequenceGenerator(object):
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
                label = sum(np.array(s))%2

            self.labels.append(label)
            # self.labels.append(sum(np.array(s)))
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
    sample = DigitSequenceGenerator(n_samples=20)
    for i in range(1):
        batch_data, batch_labels, batch_seqlen = sample.next(3)
        print(np.array(batch_data))
        print(batch_labels)
        print(batch_seqlen)