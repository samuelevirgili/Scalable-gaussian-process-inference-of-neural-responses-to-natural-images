import numpy as np
import os
import pickle


DATA_FILE = "/media/samuele/Samuele_2/Benchmark_models_for_Matthew/data2_41mixed_tr28.pkl"


class Dataset:

    def __init__(self,
                 images_train,
                 responses_train,
                 images_val,
                 responses_val,
                 images_test,
                 responses_test):

        # normalize images (mean=0, SD=1)
        # m = images_train.mean()
        # sd = images_train.std()
        # zscore = lambda img: (img - m) / sd
        # self.images_train = zscore(images_train)[...,None]
        # self.images_val = zscore(images_val)[...,None]
        # self.images_test = zscore(images_test)[...,None]
        self.images_train = images_train
        self.images_val = images_val
        self.images_test = images_test

        # normalize responses (SD=1)
        # sd = responses_train.std(axis=0)
        # sd[sd < (sd.mean() / 100)] = 1
        # def rectify_and_normalize(x):
        #     x[x < 0] = 0    # responses are non-negative; this gets rid
        #                     # of small negative numbers due to numerics
        #     return x / sd
        # self.responses_train = rectify_and_normalize(responses_train)
        # self.responses_val = rectify_and_normalize(responses_val)
        # self.responses_test = rectify_and_normalize(responses_test)
        self.responses_train = responses_train
        self.responses_val = responses_val
        self.responses_test = responses_test

        self.num_neurons = responses_train.shape[1]
        self.num_train_samples = images_train.shape[0]
        self.px_x = images_train.shape[2]
        self.px_y = images_train.shape[1]
        self.input_shape = [None, self.px_y, self.px_x, 1]
        self.minibatch_idx = 1e10
        self.train_perm = []

        self.cell_selection = None

    def get_cell_nbs(self):
        return [cell_nb for cell_nb in range(0, self.num_neurons)]

    def select_cells(self, selection):
        if selection == 'all':
            selection = None
        self.cell_selection = selection
        return

    def val(self):
        images = self.images_val
        responses = self.responses_val
        if self.cell_selection is not None:
            responses = self.responses_val[:, self.cell_selection]
        return images, responses

    def train(self):
        images = self.images_train
        responses = self.responses_train
        if self.cell_selection is not None:
            responses = responses[:, self.cell_selection]
        return images, responses

    def test(self, cell_selection=None, averages=True):
        images = self.images_test
        responses = self.responses_test
        if self.cell_selection is not None:
            responses = responses[:, :, self.cell_selection]
        if averages:
            responses = responses.mean(axis=0)
        return images, responses

    def minibatch(self, batch_size):
        if self.minibatch_idx + batch_size > self.num_train_samples:
            self.next_epoch()
        idx = self.train_perm[self.minibatch_idx + np.arange(0, batch_size)]
        self.minibatch_idx += batch_size
        return self.images_train[idx, :, :], self.responses_train[idx, :]

    def next_epoch(self):
        self.minibatch_idx = 0
        self.train_perm = np.random.permutation(self.num_train_samples)

    def save(self, data_file=DATA_FILE):
        if os.path.isfile(data_file):
            raise FileExistsError
        with open(data_file, mode='wb') as file:
            pickle.dump(self, file)
        return

    @staticmethod
    def load(data_file=DATA_FILE):
        with open(data_file, 'rb') as file:
            dataset = pickle.load(file)
        return dataset
