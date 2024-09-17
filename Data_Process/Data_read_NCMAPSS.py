import os
import pandas as pd
import numpy as np
import random

import gc

import os
import json
import logging
import sys
import h5py
import time
from sklearn import preprocessing
import torch
import matplotlib.pyplot as plt

class NCMAPSS():
    def __init__(self, data_root, window_size, stride, subsampling):

        self.window_size = window_size
        self.stride = stride
        self.subsampling = subsampling

        data_df = self._get_data(data_root, subsampling)
        self.train_x, self.train_y, self.test_x, self.test_y, self.max_rul = self._process(data_df)
        print(self.max_rul)
        print(np.array(self.train_x).shape)
        print(np.array(self.train_y).shape)
        for key, value in self.test_x.items():
            print(key)
            print(np.array(value).shape)

        self.data_save()

    def _get_data(self, path, sampling):
        path = os.path.join(path, 'N-CMAPSS', 'N-CMAPSS_DS02-006.h5')
        assert os.path.exists(path), 'data path does not exist: {:}'.format(path)
        with h5py.File(path, 'r') as hdf:
            # Development(training) set
            W_dev = np.array(hdf.get('W_dev'))  # W
            X_s_dev = np.array(hdf.get('X_s_dev'))  # X_s
            X_v_dev = np.array(hdf.get('X_v_dev'))  # X_v
            T_dev = np.array(hdf.get('T_dev'))  # T
            Y_dev = np.array(hdf.get('Y_dev'))  # RUL
            A_dev = np.array(hdf.get('A_dev'))  # Auxiliary

            # Test set
            W_test = np.array(hdf.get('W_test'))  # W
            X_s_test = np.array(hdf.get('X_s_test'))  # X_s
            X_v_test = np.array(hdf.get('X_v_test'))  # X_v
            T_test = np.array(hdf.get('T_test'))  # T
            Y_test = np.array(hdf.get('Y_test'))  # RUL
            A_test = np.array(hdf.get('A_test'))  # Auxiliary

            # Varnams
            W_var = np.array(hdf.get('W_var'))
            X_s_var = np.array(hdf.get('X_s_var'))
            X_v_var = np.array(hdf.get('X_v_var'))
            T_var = np.array(hdf.get('T_var'))
            A_var = np.array(hdf.get('A_var'))

            # from np.array to list dtype U4/U5
            W_var = list(np.array(W_var, dtype='U20'))
            X_s_var = list(np.array(X_s_var, dtype='U20'))
            X_v_var = list(np.array(X_v_var, dtype='U20'))
            T_var = list(np.array(T_var, dtype='U20'))
            A_var = list(np.array(A_var, dtype='U20'))

        W = np.concatenate((W_dev, W_test), axis=0)
        X_s = np.concatenate((X_s_dev, X_s_test), axis=0)
        X_v = np.concatenate((X_v_dev, X_v_test), axis=0)
        T = np.concatenate((T_dev, T_test), axis=0)
        Y = np.concatenate((Y_dev, Y_test), axis=0)
        A = np.concatenate((A_dev, A_test), axis=0)

        print('')
        print("number of training samples(timestamps): ", Y_dev.shape[0])
        print("number of test samples(timestamps): ", Y_test.shape[0])
        print('')
        print("W shape: " + str(W.shape))
        print("X_s shape: " + str(X_s.shape))
        print("X_v shape: " + str(X_v.shape))
        print("T shape: " + str(T.shape))
        print("Y shape: " + str(Y.shape))
        print("A shape: " + str(A.shape))

        '''
        Illusration of Multivariate time-series of condition monitoring sensors readings for Unit5 (fifth engine)

        W: operative conditions (Scenario descriptors) - ['alt', 'Mach', 'TRA', 'T2']
        X_s: measured signals - ['T24', 'T30', 'T48', 'T50', 'P15', 'P2', 'P21', 'P24', 'Ps30', 'P40', 'P50', 'Nf', 'Nc', 'Wf']
        X_v: virtual sensors - ['T40', 'P30', 'P45', 'W21', 'W22', 'W25', 'W31', 'W32', 'W48', 'W50', 'SmFan', 'SmLPC', 'SmHPC', 'phi']
        T(theta): engine health parameters - ['fan_eff_mod', 'fan_flow_mod', 'LPC_eff_mod', 'LPC_flow_mod', 'HPC_eff_mod', 'HPC_flow_mod', 'HPT_eff_mod', 'HPT_flow_mod', 'LPT_eff_mod', 'LPT_flow_mod']
        Y: RUL [in cycles]
        A: auxiliary data - ['unit', 'cycle', 'Fc', 'hs']
        '''

        df_W = pd.DataFrame(data=W, columns=W_var)
        df_Xs = pd.DataFrame(data=X_s, columns=X_s_var)
        df_Xv = pd.DataFrame(data=X_v[:, 0:2], columns=['T40', 'P30'])
        # df_T = pd.DataFrame(data=T, columns=T_var)
        df_Y = pd.DataFrame(data=Y, columns=['RUL'])
        df_A = pd.DataFrame(data=A, columns=A_var).drop(columns=['cycle', 'Fc', 'hs'])

        # Merge all the dataframes
        df_all = pd.concat([df_W, df_Xs, df_Xv, df_Y, df_A], axis=1)
        print('labels are ', df_Y)

        print("df_all",
              df_all)  # df_all = pd.concat([df_W, df_Xs, df_Xv, df_Y, df_A], axis=1).drop(columns=[ 'P45', 'W21', 'W22', 'W25', 'W31', 'W32', 'W48', 'W50', 'SmFan', 'SmLPC', 'SmHPC', 'phi', 'Fc', 'hs'])

        print("df_all.shape", df_all.shape)

        df_all_smp = df_all[::sampling]
        print("df_all_sub",
              df_all_smp)  # df_all = pd.concat([df_W, df_Xs, df_Xv, df_Y, df_A], axis=1).drop(columns=[ 'P45', 'W21', 'W22', 'W25', 'W31', 'W32', 'W48', 'W50', 'SmFan', 'SmLPC', 'SmHPC', 'phi', 'Fc', 'hs'])

        print("df_all_sub.shape", df_all_smp.shape)

        return df_all_smp

    def _process(self, df_all):
        units_index_train = [2.0, 5.0, 10.0, 16.0, 18.0, 20.0]
        units_index_test = [11.0, 14.0, 15.0]

        df_train = self.df_creator(df_all, units_index_train)
        df_test = self.df_creator(df_all, units_index_test)

        gc.collect()

        cols_normalize = df_train.columns.difference(['RUL', 'unit'])

        df_train, df_test = self._transform(df_train, df_test, cols_normalize)
        del df_all

        train_x = []
        train_y = []
        test_x = dict()
        test_y = dict()
        max_RULs = dict()

        for unit_index in units_index_train:
            label_unit = self.time_window_slicing_label_save(df_train, self.window_size, self.stride, unit_index,
                                                             sequence_cols='RUL')
            sample_unit = self.time_window_slicing_sample_save(df_train, self.window_size, self.stride, unit_index,
                                                               sequence_cols=cols_normalize)
            train_x.append(sample_unit)
            train_y.append(label_unit)
        train_x = np.concatenate(train_x, 0)
        train_y = np.concatenate(train_y, 0)

        max_RUL = max(train_y)
        train_y = train_y / max_RUL

        # combined_data = list(zip(train_x, train_y))
        #
        # random.shuffle(combined_data)
        #
        # train_x, train_y = zip(*combined_data)

        for unit_index in units_index_test:
            label_unit = self.time_window_slicing_label_save(df_test, self.window_size, self.stride, unit_index,
                                                             sequence_cols='RUL')
            sample_unit = self.time_window_slicing_sample_save(df_test, self.window_size, self.stride, unit_index,
                                                               sequence_cols=cols_normalize)
            test_x[unit_index] = sample_unit
            test_y[unit_index] = label_unit / max_RUL
            max_RULs[unit_index] = max_RUL
        return train_x, train_y, test_x, test_y, max_RULs

    def df_creator(self, df_all, units_index_train):
        train_df_lst = []
        for idx in units_index_train:
            df_train_temp = df_all[df_all['unit'] == np.float64(idx)]
            train_df_lst.append(df_train_temp)
        df_train = pd.concat(train_df_lst)
        df_train = df_train.reset_index(drop=True)

        return df_train

    def time_window_slicing_label_save(self, input_array, sequence_length, stride, index,
                                       sequence_cols='RUL'):
        '''
        ref
            for i in range(0, input_temp.shape[0] - sequence_length):
            window = input_temp[i*stride:i*stride + sequence_length, :]  # each individual window
            window_lst.append(window)
            # print (window.shape)


        '''
        # generate labels
        window_lst = []  # a python list to hold the windows

        input_temp = input_array[input_array['unit'] == index][sequence_cols].values
        num_samples = int((input_temp.shape[0] - sequence_length) / stride) + 1
        for i in range(num_samples):
            window = input_temp[i * stride:i * stride + sequence_length]  # each individual window
            window_lst.append(window)

        label_array = np.asarray(window_lst).astype(np.float32)

        return label_array[:, -1]

    def time_window_slicing_sample_save(self, input_array, sequence_length, stride, index, sequence_cols):
        '''


        '''
        # generate labels
        window_lst = []  # a python list to hold the windows

        input_temp = input_array[input_array['unit'] == index][sequence_cols].values
        print("Unit%s input array shape: " % index, input_temp.shape)
        num_samples = int((input_temp.shape[0] - sequence_length) / stride) + 1
        for i in range(num_samples):
            window = input_temp[i * stride:i * stride + sequence_length, :]  # each individual window
            window_lst.append(window)

        sample_array = np.dstack(window_lst).astype(np.float32)
        print("sample_array.shape", sample_array.shape)

        sample_array = np.transpose(sample_array, (2, 0, 1))
        print("sample_array.shape", sample_array.shape)

        return sample_array

    def _transform(self, df_train, df_test, cols_normalize):
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))

        # Normalize the training data
        df_train = df_train.rename(str, axis="columns")
        df_test = df_test.rename(str, axis="columns")

        transformed_data = min_max_scaler.fit_transform(df_train[cols_normalize])
        norm_df = pd.DataFrame(transformed_data, columns=cols_normalize,
                               index=df_train.index)
        join_df = df_train[df_train.columns.difference(cols_normalize)].join(norm_df)
        df_train = join_df.reindex(columns=df_train.columns)

        # Normalize the test data
        norm_test_df = pd.DataFrame(min_max_scaler.transform(df_test[cols_normalize]), columns=cols_normalize,
                                    index=df_test.index)
        test_join_df = df_test[df_test.columns.difference(cols_normalize)].join(norm_test_df)
        df_test = test_join_df.reindex(columns=df_test.columns)
        df_test = df_test.reset_index(drop=True)

        return df_train, df_test

    def data_save(self):
        if not os.path.exists('Processed_dataset'):
            os.mkdir('Processed_dataset')
        data_dir = os.path.join('Processed_dataset', 'NCMAPSS')
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        # condition_data_dir = os.path.join(data_dir, f'{self.data_set}')
        # if not os.path.exists(condition_data_dir):
        #     os.mkdir(condition_data_dir)

        torch.save({'samples': self.train_x, 'labels': self.train_y, 'max_ruls': self.max_rul}, f'{data_dir}/train.pt')
        torch.save({'samples': self.test_x, 'labels': self.test_y, 'max_ruls': self.max_rul}, f'{data_dir}/test.pt')


if __name__ == "__main__":
    ROOT_PATH = r"Datasets"
    data = NCMAPSS(ROOT_PATH, window_size=50, stride=1, subsampling=100)