import numpy as np
import h5py, os, pickle, torch
import time
from framework.utilities import calculate_scalar, scale
import framework.config as config
from framework.data_split_proportionally import data_split


class DataGenerator_DNN_CNN_CNN_Transformer(object):
    def __init__(self, batch_size=config.batch_size, seed=42, dataset_path=None, normalization=False, split_type=None):

        ############### data split #######################################################
        test_file = os.path.join(dataset_path, 'test_audio_id.txt')
        if not os.path.exists(test_file):
            print('data spliting... :', split_type)
            data_split(dataset_path, dir_name=split_type)

        self.batch_size = batch_size
        self.random_state = np.random.RandomState(seed)
        self.validate_random_state = np.random.RandomState(0)
        self.test_random_state = np.random.RandomState(0)

        # Load data
        load_time = time.time()

        file_path = os.path.join(dataset_path, 'training.pickle')
        print('using: ', file_path)
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        self.train_audio_ids, self.train_rates, self.train_event_label = \
            data['audio_ids'], data['rates'], data['event_label']
        self.train_x = data['x']

        file_path = os.path.join(dataset_path, 'validation.pickle')
        print('using: ', file_path)
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        self.val_audio_ids, self.val_rates, self.val_event_label = \
            data['audio_ids'], data['rates'], data['event_label']
        self.val_x = data['x']

        file_path = os.path.join(dataset_path, 'test.pickle')
        print('using test: ', file_path)
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        self.test_audio_ids, self.test_rates, self.test_event_label = \
            data['audio_ids'], data['rates'], data['event_label']
        self.test_x = data['x']

        ############################################################################################

        print('Loading data time: {:.3f} s'.format(time.time() - load_time))

        print('Split development data to {} training and {} '
              'validation data and {} '
              'test data. '.format(len(self.train_audio_ids),
                                         len(self.val_audio_ids),
                                   len(self.test_audio_ids)))

        self.normal = normalization
        if self.normal:
            (self.mean, self.std) = calculate_scalar(self.train_x)


    def load_x(self, audio_ids):
        if os.path.exists(r'D:\Yuanbo\Code\GPULab\Code\UCL\meta_data'):
            root = r'D:\Yuanbo\Code\GPULab\Code\UCL\meta_data'
        elif os.path.exists(r'E:\Yuanbo\UCL\DeLTA\meta_data'):
            root = r'E:\Yuanbo\UCL\DeLTA\meta_data'
        elif os.path.exists(r'D:\Yuanbo\Code\UCL\meta_data'):
            root = r'D:\Yuanbo\Code\UCL\meta_data'
        elif os.path.exists('/project_antwerp/yuanbo/Code/UCL/meta_data'):
            root = '/project_antwerp/yuanbo/Code/UCL/meta_data'
        elif os.path.exists('/project_scratch/yuanbo/Code/UCL/meta_data'):
            root = '/project_scratch/yuanbo/Code/UCL/meta_data'

        x_list = []
        for each_id in audio_ids:
            idfile = os.path.join(root, 'DeLTA_mp3_boost_8dB_mel64', each_id.replace('.mp3', '.npy'))
            x_list.append(np.load(idfile))
        return np.array(x_list)


    def load_rate_event_id(self, sub_set, dataset_path):
        audio_id_file = os.path.join(dataset_path, sub_set + '_audio_id.txt')
        rate_file = os.path.join(dataset_path, sub_set + '_annoyance_rate.txt')
        event_label_file = os.path.join(dataset_path, sub_set + '_sound_source.txt')

        audio_ids = []
        with open(audio_id_file, 'r') as f:
            for line in f.readlines():
                part = line.split('\n')[0]
                if part:
                    audio_ids.append(part)
        rates = np.loadtxt(rate_file)[:, None]
        event_label = np.loadtxt(event_label_file)
        return audio_ids, rates, event_label

    def generate_train(self):
        audios_num = len(self.train_audio_ids)
        audio_indexes = [i for i in range(audios_num)]

        self.random_state.shuffle(audio_indexes)

        iteration = 0
        pointer = 0

        while True:
            if pointer >= audios_num:
                pointer = 0
                self.random_state.shuffle(audio_indexes)

            # Get batch indexes
            batch_audio_indexes = audio_indexes[pointer: pointer + self.batch_size]
            pointer += self.batch_size

            iteration += 1
            batch_x = self.train_x[batch_audio_indexes]
            batch_y = self.train_rates[batch_audio_indexes]
            batch_y_event = self.train_event_label[batch_audio_indexes]
            if self.normal:
                batch_x = self.transform(batch_x)

            yield batch_x, batch_y, batch_y_event


    def generate_validate(self, data_type, max_iteration=None):
        audios_num = len(self.val_audio_ids)
        audio_indexes = [i for i in range(audios_num)]

        self.validate_random_state.shuffle(audio_indexes)

        print('Number of {} audios in {}'.format(len(audio_indexes), data_type))

        iteration = 0
        pointer = 0

        while True:
            if iteration == max_iteration:
                break

            # Reset pointer
            if pointer >= audios_num:
                break

            batch_audio_indexes = audio_indexes[pointer: pointer + self.batch_size]
            pointer += self.batch_size

            iteration += 1
            batch_x = self.val_x[batch_audio_indexes]
            batch_y = self.val_rates[batch_audio_indexes]
            batch_y_event = self.val_event_label[batch_audio_indexes]

            if self.normal:
                batch_x = self.transform(batch_x)

            yield batch_x, batch_y, batch_y_event


    def generate_testing(self, data_type, max_iteration=None):
        audios_num = len(self.test_audio_ids)
        audio_indexes = [i for i in range(audios_num)]

        self.test_random_state.shuffle(audio_indexes)

        print('Number of {} audios in {}'.format(len(audio_indexes), data_type))

        iteration = 0
        pointer = 0

        while True:
            if iteration == max_iteration:
                break

            # Reset pointer
            if pointer >= audios_num:
                break

            batch_audio_indexes = audio_indexes[pointer: pointer + self.batch_size]
            pointer += self.batch_size

            iteration += 1
            batch_x = self.test_x[batch_audio_indexes]
            batch_y = self.test_rates[batch_audio_indexes]
            batch_y_event = self.test_event_label[batch_audio_indexes]

            if self.normal:
                batch_x = self.transform(batch_x)

            yield batch_x, batch_y, batch_y_event


    def transform(self, x):
        """Transform data.

        Args:
          x: (batch_x, seq_len, freq_bins) | (seq_len, freq_bins)

        Returns:
          Transformed data.
        """

        return scale(x, self.mean, self.std)


class DataGenerator_HGRL(object):
    def __init__(self, emb_dim, seed=42, dataset_path=None, normalization=False, split_type=None):

        ############### data split #######################################################
        test_file = os.path.join(dataset_path, 'test_audio_id.txt')
        if not os.path.exists(test_file):
            print('data spliting... :', split_type)
            data_split(dataset_path, dir_name=split_type)

        self.batch_size = config.batch_size
        self.random_state = np.random.RandomState(seed)
        self.validate_random_state = np.random.RandomState(0)
        self.test_random_state = np.random.RandomState(0)

        # Load data
        load_time = time.time()

        file_path = os.path.join(dataset_path, 'training.pickle')
        print('using: ', file_path)
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        self.train_audio_ids, self.train_rates, self.train_event_label = \
            data['audio_ids'], data['rates'], data['event_label']
        self.train_x = data['x']


        file_path = os.path.join(dataset_path, 'validation.pickle')
        print('using: ', file_path)
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        self.val_audio_ids, self.val_rates, self.val_event_label = \
            data['audio_ids'], data['rates'], data['event_label']
        self.val_x = data['x']

        file_path = os.path.join(dataset_path, 'test.pickle')
        print('using: ', file_path)
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        self.test_audio_ids, self.test_rates, self.test_event_label = \
            data['audio_ids'], data['rates'], data['event_label']
        self.test_x = data['x']

        ################################### map and sort labels ######################################
        # print(self.train_event_label.shape)
        # (2200, 24)
        self.train_event_label, self.train_coarse_level_subject_labels = self.sort_map_labels(
            self.train_event_label)

        self.val_event_label, self.val_coarse_level_subject_labels = self.sort_map_labels(
            self.val_event_label)

        self.test_event_label, self.test_coarse_level_subject_labels = self.sort_map_labels(
            self.test_event_label)

        ##############################################################################################
        self.graph_event_list = []
        self.output_all_pickle_file = os.path.join(dataset_path, 'blank_all_event24_semantic7_rate1' + '_ed'
                                                   + str(emb_dim) + '.pickle')

        print('loading output_all_pickle_file ...')
        with open(self.output_all_pickle_file, 'rb') as tf:
            pkl = pickle.load(tf)
        for key, value in pkl.items():
            self.graph_event_list.append(value)


        ############################################################################################
        #  Split development data to 2200 training and 445 validation data.
        print('Loading data time: {:.3f} s'.format(time.time() - load_time))

        print('Split development data to {} training and {} '
              'validation data and {} '
              'test data. '.format(len(self.train_audio_ids),
                                   len(self.val_audio_ids),
                                   len(self.test_audio_ids)))

        self.normal = normalization
        if self.normal:
            (self.mean, self.std) = calculate_scalar(self.train_x)


    def sort_map_labels(self, source_labels):
        sort_object_event_labels = []
        coarse_level_subject_labels = []

        for i in range(len(source_labels)):
            row = source_labels[i]
            # print(row)
            new_row = np.zeros_like(row)

            new_coarse_subject_row = np.zeros(len(config.subject_labels))
            # print(new_coarse_subject_row)

            for num, each in enumerate(list(row)):
                if each:
                    new_row[config.source_to_sort_indices[num]] = 1

            if sum(new_row[:7]):
                new_coarse_subject_row[0] = 1
            if sum(new_row[7:9]):
                new_coarse_subject_row[1] = 1
            if sum(new_row[9:11]):
                new_coarse_subject_row[2] = 1
            if sum(new_row[11:16]):
                new_coarse_subject_row[3] = 1
            if sum(new_row[16:18]):
                new_coarse_subject_row[4] = 1
            if sum(new_row[18:20]):
                new_coarse_subject_row[5] = 1
            if sum(new_row[20:24]):
                new_coarse_subject_row[6] = 1

            sort_object_event_labels.append(new_row)
            coarse_level_subject_labels.append(new_coarse_subject_row)

        sort_object_event_labels = np.stack(sort_object_event_labels)
        coarse_level_subject_labels = np.stack(coarse_level_subject_labels)
        return sort_object_event_labels, coarse_level_subject_labels


    def load_x(self, audio_ids):
        if os.path.exists(r'D:\Yuanbo\Code\GPULab\Code\UCL\meta_data'):
            root = r'D:\Yuanbo\Code\GPULab\Code\UCL\meta_data'
        elif os.path.exists(r'E:\Yuanbo\UCL\DeLTA\meta_data'):
            root = r'E:\Yuanbo\UCL\DeLTA\meta_data'
        elif os.path.exists(r'D:\Yuanbo\Code\UCL\meta_data'):
            root = r'D:\Yuanbo\Code\UCL\meta_data'
        elif os.path.exists('/project_antwerp/yuanbo/Code/UCL/meta_data'):
            root = '/project_antwerp/yuanbo/Code/UCL/meta_data'
        elif os.path.exists('/project_scratch/yuanbo/Code/UCL/meta_data'):
            root = '/project_scratch/yuanbo/Code/UCL/meta_data'

        x_list = []
        for each_id in audio_ids:
            idfile = os.path.join(root, 'DeLTA_mp3_boost_8dB_mel64', each_id.replace('.mp3', '.npy'))
            x_list.append(np.load(idfile))
        return np.array(x_list)


    def load_rate_event_id(self, split_type, sub_set):
        data_path = os.path.join(config.root, split_type)
        audio_id_file = os.path.join(data_path, sub_set + '_audio_id.txt')
        rate_file = os.path.join(data_path, sub_set + '_annoyance_rate.txt')
        event_label_file = os.path.join(data_path, sub_set + '_sound_source.txt')

        audio_ids = []
        with open(audio_id_file, 'r') as f:
            for line in f.readlines():
                part = line.split('\n')[0]
                if part:
                    audio_ids.append(part)
        rates = np.loadtxt(rate_file)[:, None]
        event_label = np.loadtxt(event_label_file)
        return audio_ids, rates, event_label

    def generate_train(self):
        audios_num = len(self.train_audio_ids)
        audio_indexes = [i for i in range(audios_num)]

        self.random_state.shuffle(audio_indexes)

        iteration = 0
        pointer = 0

        while True:
            if pointer >= audios_num:
                pointer = 0
                self.random_state.shuffle(audio_indexes)

            # Get batch indexes
            batch_audio_indexes = audio_indexes[pointer: pointer + self.batch_size]
            pointer += self.batch_size

            iteration += 1
            batch_graph = [self.graph_event_list[each] for each in batch_audio_indexes]
            batch_x = self.train_x[batch_audio_indexes]
            batch_y = self.train_rates[batch_audio_indexes]
            batch_y_event = self.train_event_label[batch_audio_indexes]
            batch_y_semantic7 = self.train_coarse_level_subject_labels[batch_audio_indexes]

            if self.normal:
                batch_x = self.transform(batch_x)

            yield batch_x, batch_y, batch_y_event, batch_graph, batch_y_semantic7


    def generate_validate(self, data_type, max_iteration=None):
        audios_num = len(self.val_audio_ids)
        audio_indexes = [i for i in range(audios_num)]

        self.validate_random_state.shuffle(audio_indexes)

        print('Number of {} audios in {}'.format(len(audio_indexes), data_type))

        iteration = 0
        pointer = 0

        while True:
            if iteration == max_iteration:
                break

            # Reset pointer
            if pointer >= audios_num:
                break

            batch_audio_indexes = audio_indexes[pointer: pointer + self.batch_size]
            pointer += self.batch_size

            iteration += 1
            batch_graph = [self.graph_event_list[each] for each in batch_audio_indexes]
            batch_x = self.val_x[batch_audio_indexes]
            batch_y = self.val_rates[batch_audio_indexes]
            batch_y_event = self.val_event_label[batch_audio_indexes]
            batch_y_semantic7 = self.val_coarse_level_subject_labels[batch_audio_indexes]

            if self.normal:
                batch_x = self.transform(batch_x)

            yield batch_x, batch_y, batch_y_event, batch_graph, batch_y_semantic7


    def generate_testing(self, data_type, max_iteration=None):
        audios_num = len(self.test_audio_ids)
        audio_indexes = [i for i in range(audios_num)]

        self.test_random_state.shuffle(audio_indexes)

        print('Number of {} audios in {}'.format(len(audio_indexes), data_type))

        iteration = 0
        pointer = 0

        while True:
            if iteration == max_iteration:
                break

            # Reset pointer
            if pointer >= audios_num:
                break

            batch_audio_indexes = audio_indexes[pointer: pointer + self.batch_size]
            pointer += self.batch_size

            iteration += 1
            batch_graph = [self.graph_event_list[each] for each in batch_audio_indexes]
            batch_x = self.test_x[batch_audio_indexes]
            batch_y = self.test_rates[batch_audio_indexes]
            batch_y_event = self.test_event_label[batch_audio_indexes]
            batch_y_semantic7 = self.test_coarse_level_subject_labels[batch_audio_indexes]

            if self.normal:
                batch_x = self.transform(batch_x)

            yield batch_x, batch_y, batch_y_event, batch_graph, batch_y_semantic7


    def transform(self, x):
        """Transform data.

        Args:
          x: (batch_x, seq_len, freq_bins) | (seq_len, freq_bins)

        Returns:
          Transformed data.
        """

        return scale(x, self.mean, self.std)


