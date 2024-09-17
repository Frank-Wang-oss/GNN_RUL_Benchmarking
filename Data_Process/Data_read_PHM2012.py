import os
import pandas as pd
import numpy as np
import random
import torch




class PHM_2012():
    def __init__(self, data_root, condition_no):
        self.RUL_dict = {'Bearing1_1':0,'Bearing1_2':0,
                'Bearing2_1':0,'Bearing2_2':0,
                'Bearing3_1':0,'Bearing3_2':0,
                'Bearing1_3':573,'Bearing1_4':33.9,'Bearing1_5':161,'Bearing1_6':146,'Bearing1_7':757,
                'Bearing2_3':753,'Bearing2_4':139,'Bearing2_5':309,'Bearing2_6':129,'Bearing2_7':58,
                'Bearing3_3':82}
        self.train_test_split = {
            'Condition1_train':['Bearing1_1','Bearing1_2'],
            'Condition1_test': ['Bearing1_3', 'Bearing1_4','Bearing1_5', 'Bearing1_6','Bearing1_7'],
            'Condition2_train': ['Bearing2_1', 'Bearing2_2'],
            'Condition2_test': ['Bearing2_3', 'Bearing2_4', 'Bearing2_5', 'Bearing2_6', 'Bearing2_7'],
            'Condition3_train': ['Bearing3_1', 'Bearing3_2'],
            'Condition3_test': ['Bearing3_3']
        }
        self.RUL_condition_no = condition_no
        x_cond_dict, y_cond_dict, self.max_rul = self._get_data(data_root)
        self.train_x, self.train_y, self.test_x, self.test_y = self._process(x_cond_dict, y_cond_dict)

        print(np.array(self.train_x).shape)
        print(np.array(self.train_y).shape)
        for key, value in self.test_x.items():
            print(key)
            print(np.array(value).shape)
        self.data_save()


    def _get_data(self, path):
        path = os.path.join(path, 'PHM_2012_Bearing_Datasets')
        assert os.path.exists(path), 'data path does not exist: {:}'.format(path)
        x = dict()
        y = dict()
        max_rul_bearings = dict()

        for path_train_test in ['Learning_set/', 'Test_set/']:

            traintestFolderPath = os.path.join(path, path_train_test)
            train_lists = self.train_test_split[f'Condition{self.RUL_condition_no}_train']
            test_lists = self.train_test_split[f'Condition{self.RUL_condition_no}_test']
            bearings_names = os.listdir(traintestFolderPath)
            bearings_names.sort()
            for bearing_no in bearings_names:
                if bearing_no in train_lists+test_lists:
                    print(f'Start processing {bearing_no} in {path_train_test}')

                    Bearing_PATH = os.path.join(traintestFolderPath, bearing_no)
                    file_names = os.listdir(Bearing_PATH)
                    file_names.sort()
                    samples = []
                    labels = []
                    for file_name in file_names:
                        if 'acc' in file_name:
                            sample_path = os.path.join(Bearing_PATH, file_name)
                            df = pd.read_csv(sample_path,header=None)
                            sample = np.array(df.loc[:,4])
                            # print(sample)
                            samples.append(sample)

                    samples = np.stack(samples)
                    num_sample_bearing = len(samples)
                    RUL_last_sample = self.RUL_dict[bearing_no]
                    for idx in range(num_sample_bearing):
                        real_RUL = num_sample_bearing - idx + RUL_last_sample
                        labels.append(real_RUL)
                    max_rul = np.max(labels)
                    labels = labels/max_rul


                    x[bearing_no] = samples
                    y[bearing_no] = labels
                    max_rul_bearings[bearing_no] = max_rul

        return x,y,max_rul_bearings



    def _process(self, x, y):
        train_x_cond = []
        train_y_cond = []

        train_lists = self.train_test_split[f'Condition{self.RUL_condition_no}_train']
        test_lists = self.train_test_split[f'Condition{self.RUL_condition_no}_test']


        test_x_cond = dict()
        test_y_cond = dict()

        for bearing_i, bearing_i_signal in x.items():

            if bearing_i in train_lists:
                train_x_cond.append(x[bearing_i])
                train_y_cond.append(y[bearing_i])
            elif bearing_i in test_lists:
                test_x_cond[bearing_i] = x[bearing_i]
                test_y_cond[bearing_i] = y[bearing_i]


        train_x = np.concatenate(train_x_cond,0)
        train_y = np.concatenate(train_y_cond,0)

        max_train, min_train = np.max(train_x,0),np.min(train_x,0)
        train_x = (train_x-min_train)/(max_train-min_train)

        for key, test_x in test_x_cond.items():
            test_x = (test_x-min_train)/(max_train-min_train)
            test_x_cond[key] = test_x

        combined_data = list(zip(train_x, train_y))

        random.shuffle(combined_data)

        train_x, train_y = zip(*combined_data)

        return np.array(train_x), np.array(train_y), test_x_cond, test_y_cond

    def data_save(self):
        if not os.path.exists('Processed_dataset'):
            os.mkdir('Processed_dataset')
        data_dir = os.path.join('Processed_dataset','PHM2012')
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        condition_data_dir = os.path.join(data_dir, f'Condition_{self.RUL_condition_no}')
        if not os.path.exists(condition_data_dir):
            os.mkdir(condition_data_dir)

        torch.save({'samples':self.train_x,'labels':self.train_y,'max_ruls':self.max_rul},f'{condition_data_dir}/train.pt')
        torch.save({'samples':self.test_x,'labels':self.test_y,'max_ruls':self.max_rul},f'{condition_data_dir}/test.pt')
if __name__ == "__main__":

    ROOT_PATH = r"Datasets"
    for i in range(1,4):
        print(f'Condition {i} is being processed')
        data = PHM_2012(ROOT_PATH, condition_no=i)