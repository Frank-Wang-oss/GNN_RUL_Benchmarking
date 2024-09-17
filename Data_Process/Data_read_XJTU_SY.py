import os
import pandas as pd
import numpy as np
import random
import torch




class XJTU_SY():
    def __init__(self, data_root, condition_no, augment, downsampling, augment_ratio):

        RUL_condition_bearing = [[123, 161, 158, 122, 52], [491, 161, 533, 42, 339], [2538, 2496, 371, 1515, 114]]
        Condition_folder_path = ["35Hz12kN", "37.5Hz11kN", "40Hz10kN"]

        self.condition_no = condition_no
        self.RUL_condition = RUL_condition_bearing[condition_no-1]
        self.condition_folder = Condition_folder_path[condition_no-1]
        self.SamplingRatio = downsampling
        self.TIMESTEP = 32768

        x_cond_dict, y_cond_dict,self.max_rul = self._get_data(condition_no, data_root, augment, augment_ratio)
        self.train_x, self.train_y, self.test_x, self.test_y = self._process(x_cond_dict, y_cond_dict)
        self.data_save()
    def read_signal_file_to_array_by_path(self, path):
        pd_data = pd.read_csv(path)
        # print(pd_data)
        horizontal_signal = pd_data["Horizontal_vibration_signals"][::self.SamplingRatio]
        # print(horizontal_signal.shape)
        return horizontal_signal

    def read_augment_signal_file_to_array_by_path(self, path, ratio=8):
        pd_data = pd.read_csv(path)
        # print(pd_data)
        sample_base_index = np.arange(0,self.TIMESTEP,self.SamplingRatio)

        horizontal_signal = pd_data["Horizontal_vibration_signals"].reshape(-1, 1)
        augment_step_range = self.SamplingRatio // ratio

        augment_dataset = []

        for step in range(0,augment_step_range):
            sample_index = sample_base_index + step
            augment_sample = horizontal_signal[sample_index,:]
            augment_dataset.append(augment_sample)

        return augment_dataset

    def _get_data(self, conditionVal, path, augment, ratio):
        path = os.path.join(path, 'XJTU-SY_Bearing_Datasets')
        assert os.path.exists(path), 'data path does not exist: {:}'.format(path)
        x = dict()
        y = dict()
        max_rul_bearing = dict()

        conditionFolderPath = os.path.join(path, self.condition_folder)
        for subSampleVal in range(1, 6):
            x_con = []
            y_con = []
            bearingSampleFolderPath = os.path.join(conditionFolderPath,
                                                   "Bearing%d_%d" % (conditionVal, subSampleVal))
            max_RUL = self.RUL_condition[subSampleVal - 1]

            print("# MAX SAMPLE RUL :", max_RUL)
            print("@ Reading CSV of Bearing ", conditionVal, subSampleVal)

            startMinute = 1
            endMinutes = max_RUL

            # add sample to data set
            for i in range(startMinute, endMinutes + 1):

                sampleCSVPath = os.path.join(bearingSampleFolderPath, "%d.csv" % (i))


                if augment:
                    augment_data = self.read_augment_signal_file_to_array_by_path(sampleCSVPath, ratio=ratio)
                    for augment_sample in augment_data:
                        x_con.append(augment_sample)
                        y_con.append((endMinutes - i)/endMinutes)

                else:
                    np_data = self.read_signal_file_to_array_by_path(sampleCSVPath)
                    x_con.append(np_data)
                    y_con.append((endMinutes - i) / endMinutes)
            x_con = np.stack(x_con,0)
            y_con = np.stack(y_con,0)

            x[subSampleVal] = x_con
            y[subSampleVal] = y_con
            max_rul_bearing[subSampleVal] = max_RUL


        # print(x.keys())
        return x,y,max_rul_bearing



    def _process(self, x, y):
        train_x_cond = dict()
        train_y_cond = dict()

        test_x_cond = dict()
        test_y_cond = dict()
        for bearing_i in x.keys():
            print('Testing bearing is', bearing_i)
            ## bearing_i is the i-th bearing for evaluation
            train_x = []
            train_y = []
            test_x = []
            test_y = []

            for bearing_j, bearing_j_signal in x.items():
                if bearing_i == bearing_j:

                    test_x.append(bearing_j_signal)
                    test_y.append(y[bearing_j])
                else:
                    print('Reading training bearing of', bearing_j)
                    # print('training bearing size is', bearing_j_signal.shape)

                    train_x.append(bearing_j_signal)
                    train_y.append(y[bearing_j])

            train_x = np.concatenate(train_x,0)
            train_y = np.concatenate(train_y,0)
            test_x = np.concatenate(test_x,0)
            test_y = np.concatenate(test_y,0)

            max_train, min_train = np.max(train_x,0),np.min(train_x,0)

            train_x = (train_x-min_train)/(max_train-min_train)
            test_x = (test_x-min_train)/(max_train-min_train)

            combined_data = list(zip(train_x, train_y))

            random.shuffle(combined_data)

            train_x, train_y = zip(*combined_data)

            train_x_cond[bearing_i] = train_x
            train_y_cond[bearing_i] = train_y
            test_x_cond[bearing_i] = test_x
            test_y_cond[bearing_i] = test_y
        return train_x_cond, train_y_cond, test_x_cond, test_y_cond


    def data_save(self):
        if not os.path.exists('Processed_dataset'):
            os.mkdir('Processed_dataset')
        data_dir = os.path.join('Processed_dataset','XJTU_SY')
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        condition_data_dir = os.path.join(data_dir, f'Condition_{self.condition_no}')
        if not os.path.exists(condition_data_dir):
            os.mkdir(condition_data_dir)
        for key, value in self.train_x.items():
            Testing_bearing_data_dir = os.path.join(condition_data_dir, f'Testing_bearing_{key}')
            if not os.path.exists(Testing_bearing_data_dir):
                os.mkdir(Testing_bearing_data_dir)
            torch.save({'samples':self.train_x[key],'labels':self.train_y[key],'max_ruls':self.max_rul[key]},f'{Testing_bearing_data_dir}/train.pt')
            torch.save({'samples':self.test_x[key],'labels':self.test_y[key],'max_ruls':self.max_rul[key]},f'{Testing_bearing_data_dir}/test.pt')


if __name__ == "__main__":

    ROOT_PATH = r"Datasets"
    for i in range(1,4):
        data = XJTU_SY(ROOT_PATH, condition_no=i, augment = False, downsampling = 1, augment_ratio = 4)