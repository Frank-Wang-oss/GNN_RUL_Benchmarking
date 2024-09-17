import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms

from sklearn.model_selection import train_test_split

import os
import numpy as np
import random


class Load_Dataset(Dataset):
    def __init__(self, X_train,y_train, normalize):
        super(Load_Dataset, self).__init__()

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        if isinstance(X_train, np.ndarray):
            X_train = torch.from_numpy(X_train)
            y_train = torch.from_numpy(y_train)

        if len(X_train.shape) < 3:
            X_train = X_train.unsqueeze(2)

        if X_train.shape.index(min(X_train.shape[1], X_train.shape[2])) != 1:  # make sure the Channels in second dim
            X_train = X_train.permute(0, 2, 1)
        if len(y_train.shape) == 1:
            y_train = y_train.unsqueeze(-1)

        self.x_data = X_train
        self.y_data = y_train

        self.num_channels = X_train.shape[1]

        if normalize:
            # Assume datashape: num_samples, num_channels, seq_length
            data_mean = torch.FloatTensor(self.num_channels).fill_(0).tolist()  # assume min= number of channels
            data_std = torch.FloatTensor(self.num_channels).fill_(1).tolist()  # assume min= number of channels
            data_transform = transforms.Normalize(mean=data_mean, std=data_std)
            self.transform = data_transform
        else:
            self.transform = None

        self.len = X_train.shape[0]
        print('Dataset size ', self.x_data.size())

    def __getitem__(self, index):
        if self.transform is not None:
            output = self.transform(self.x_data[index].view(self.num_channels, -1, 1))
            self.x_data[index] = output.view(self.x_data[index].shape)

        return self.x_data[index].float(), self.y_data[index].float()

    def __len__(self):
        return self.len


def data_generator(data_path, dataset_configs, hparams):
    # loading path
    train_dataset = torch.load(os.path.join(data_path, "train.pt"))
    test_dataset = torch.load(os.path.join(data_path, "test.pt"))
    train_x = train_dataset['samples']
    train_y = train_dataset['labels']
    max_RUL = train_dataset['max_ruls']

    test_x = test_dataset['samples']
    test_y = test_dataset['labels']

    batch_size = hparams["batch_size"]

    # Loading training datasets
    # print()
    train_dataset_processed = Load_Dataset(train_x, train_y, dataset_configs.normalize)

    # Dataloaders
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset_processed, batch_size=batch_size,
                                               shuffle=dataset_configs.shuffle, drop_last=dataset_configs.drop_last, num_workers=0)


    # Loading training datasets
    if isinstance(test_x, dict):
        test_loader = dict()
        for key, value in test_x.items():
            test_dataset_processed_i = Load_Dataset(test_x[key], test_y[key], dataset_configs.normalize)
            test_loader_i = torch.utils.data.DataLoader(dataset=test_dataset_processed_i, batch_size=batch_size,
                                                      shuffle=False, drop_last=dataset_configs.drop_last, num_workers=0)
            test_loader[key] = test_loader_i
    else:
        test_dataset = Load_Dataset(test_x, test_y, dataset_configs.normalize)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
                                                  shuffle=False, drop_last=dataset_configs.drop_last, num_workers=0)
    return train_loader, test_loader, max_RUL
