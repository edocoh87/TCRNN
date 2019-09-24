import numpy as np

from DataGenerator import DataGenerator

def to_onehot(arr, vec_len):
    res = np.zeros((len(arr), vec_len))
    res[np.arange(len(arr)), arr] = 1.0
    return res

def seq_to_str(seq):
    _seq = str([int(_[0]) for _ in seq])
    return ''.join(_seq[1:-1].split(', '))

def seq_has_anomaly(seq, anomaly):
    return anomaly in seq_to_str(seq)

class AnomalySequenceGenerator(DataGenerator):
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
    def __init__(self, n_samples=1000, max_seq_len=30, min_seq_len=10,
                 max_value=9, anomaly_sequence="1111", **kwargs):
        self.data = []
        self.labels = []
        self.seqlen = []
        for i in range(n_samples):
            # Random sequence length
            _seqlen = np.random.randint(min_seq_len, max_seq_len)
            # Monitor sequence length for TensorFlow dynamic calculation
            self.seqlen.append(_seqlen)
            # Add a random or linear int sequence (50% prob)
            s = [[float(np.random.randint(0, max_value))]
                 for i in range(_seqlen)]
            
            if np.random.random() < .5:
                start_idx = np.random.randint(0, _seqlen-len(anomaly_sequence))
                for i, d in enumerate(anomaly_sequence):
                    s[start_idx+i][0] = int(d)
            
            if seq_has_anomaly(s, anomaly_sequence):
                label = 1
            else:
                label = 0

            # Pad sequence for dimension consistency
            s += [[0.] for i in range(max_seq_len - _seqlen)]
            self.data.append(s)
            

            self.labels.append(label)
            # print(self.labels)
            # self.labels.append(sum(np.array(s)))
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
    sample = AnomalySequenceGenerator(n_samples=2000, mode='anmly', min_seq_len=10, max_seq_len=50)
    print(np.mean(sample.labels, 0))
    # exit()
    # for i in range(1):
    #     batch_data, batch_labels, batch_seqlen = sample.next(3)
    #     print(batch_labels)
        # print(np.array(batch_data))
        # print(batch_labels)
        # print(batch_seqlen)
