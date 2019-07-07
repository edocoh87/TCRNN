import os
import pandas as pd
import numpy as np

class SanDiskGenerator(object):
    def __init__(self, path='../SanDisk/', do_standardize=True, take_last_k_cycles=-1, train=True, n_features=11):

        self.path = path
        
        if train:
            prefix = 'tr_'
        else:
            prefix = 'test_'

        df_pos = pd.read_csv(os.path.join(self.path, prefix + 'pos.csv'))
        df_neg = pd.read_csv(os.path.join(self.path, prefix + 'neg.csv'))
        df_pos = df_pos.drop(columns=['Unnamed: 0', 'PC', 'DUT', 'Bank', 'BLK', 'WL', 'Str'])
        df_neg = df_neg.drop(columns=['Unnamed: 0', 'PC', 'DUT', 'Bank', 'BLK', 'WL', 'Str'])
        df_pos = df_pos.fillna(0) # remove NaN values.
        df_neg = df_neg.fillna(0)

        n_cycles = int(df_neg.shape[1] / n_features)

        # check where is the first cycle that the prog_status_cyc is '1' (which means failure)...
        df_neg_status_prog = np.array([df_neg['Prog_Status_cyc_{}'.format(i)].values for i in range(1, n_cycles + 1)]).transpose()

        # generate sequence lengths, for the failed sequences it's computed above and for the 
        # positive sequences it's randomized.
        np.random.seed(0)
        self.seqlen = np.concatenate((np.argmax(df_neg_status_prog, axis=1),
                                    np.random.randint(low=1, high=n_cycles + 1, size=len(df_pos))))

        # take each row and convert it from n_cycles*n_features to an array with shape (n_cycles, n_features)
        data_neg = [df_neg.iloc[i].values.reshape(n_cycles, n_features) for i in range(len(df_neg))]
        data_pos = [df_pos.iloc[i].values.reshape(n_cycles, n_features) for i in range(len(df_pos))]

        # stack the positive and negative exampels and create their labels.
        self.data = np.array(data_neg + data_pos)
        self.labels = np.array([[0, 1]]*len(data_neg) + [[1, 0]]*len(data_pos))

        # we must shuffle the data since it's all the negative followed by all the positive...
        p = np.random.permutation(self.data.shape[0])
        self.data = self.data[p]
        self.labels = self.labels[p]
        self.seqlen = self.seqlen[p]

        self.batch_id = 0
    
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
        batch_data = (self.data[self.batch_id:end_idx])
        batch_labels = self.labels[self.batch_id:end_idx]
        batch_seqlen = (self.seqlen[self.batch_id:end_idx])
        self.batch_id = end_idx
        return batch_data, batch_labels, batch_seqlen


if __name__=='__main__':
    sample = SanDiskGenerator()
    # sample_test = PointCloudGenerator('../PointClouds/ModelNet40_cloud.h5', mode='test')
    data, labels, seqlens = sample.next(1)
    print(data)
    # print(len(sample_test.data))
    # for i in range(5):
    #     batch_data, batch_labels = sample.next(10)
        # print(batch_data.shape)
        # print(batch_labels.shape)