import h5py
from glob import glob
from os.path import join
import numpy as np
import pandas as pd
from utils import *
from pdb import set_trace as trace
import time
import threading
import multiprocessing

POS_PATH = '/specific/netapp5_2/gamir/achiya/Sandisk/new_data/PC3/split/'

class ThreadPreProcessWorker(threading.Thread):
    def __init__(self, queue, batch_size, files, n_features, pos_file=None, neg_percentage=0.9, take_last_k_cycles=-1,
        pos_replacement=True, filter_pos=False):
        threading.Thread.__init__(self, daemon=True)
        self.queue = queue
        self.batches_count = 0
        self.last_index = 0
        self.batch_size = batch_size
        self.take_last_k_cycles = take_last_k_cycles
        self.n_features = n_features
        self.files = files
        self.permutation = np.random.permutation(len(self.files))
        self.data = pd.DataFrame({'PC': []})
        self.is_finished = False
        self.pos_df = None
        if pos_file:
            self.pos_df = pd.read_csv(pos_file, index_col=False)
            self.pos_df = filter_short_strings(self.pos_df, self.n_features, self.take_last_k_cycles)
        self.neg_percentage = neg_percentage
        self.pos_replacement = pos_replacement
        self.filter_pos = filter_pos

    def load_batch(self):
        if not self.pos_replacement and self.pos_df is not None:
            self.data = self.pos_df
            self.pos_df = None
        while len(self.data) < self.batch_size and self.last_index < len(self.permutation):
            new_data = (pd.read_csv(self.files[self.permutation[self.last_index]], index_col=False))
            self.last_index += 1
            if self.filter_pos:
                new_data = new_data[new_data['Prog_Status_cyc_50'] == 0]
            new_data = filter_short_strings(new_data, self.n_features, self.take_last_k_cycles)
            self.data = pd.concat([self.data, new_data], sort=False)
        if self.last_index >= len(self.permutation):
            self.is_finished = True
        if self.pos_df is not None:
            batch = self.data.iloc[:int(self.batch_size * self.neg_percentage)]
            self.data = self.data.iloc[int(self.batch_size * self.neg_percentage):]
            pos_batch = self.pos_df.sample(n=self.batch_size - len(batch))
            batch = pd.concat([batch, pos_batch])
        else:
            batch = self.data.iloc[:self.batch_size]
            self.data = self.data.iloc[self.batch_size:]
        preprocessed_batch = preprocess_batch(batch, self.n_features, self.take_last_k_cycles)
        return preprocessed_batch

    def run(self):
        while not self.is_finished:
            self.queue.put(self.load_batch())
