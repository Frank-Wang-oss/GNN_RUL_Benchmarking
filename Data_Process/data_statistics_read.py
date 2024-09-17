import torch
import os
import numpy as np
root_path = 'Processed_dataset'

# for i in range(1,4):
#     data_path = os.path.join(root_path, 'PHM2012', f'Condition_{i}')
#     train = torch.load(os.path.join(data_path, 'train.pt'))
#     test = torch.load(os.path.join(data_path, 'test.pt'))
#     train_data = train['samples']
#     test_data = test['samples']
#     max_ruls = test['max_ruls']
#     print(f'Condition_{i}')
#     print('Training', np.array(train_data).shape)
#     for key, value in test_data.items():
#         print(key)
#         print(np.array(value).shape)
#         print(max_ruls[key])
#
#     print('='*20)


for i in range(1,4):
    data_path = os.path.join(root_path, 'XJTU_SY', f'Condition_{i}')
    for j in range(1,6):
        train = torch.load(os.path.join(data_path, f'Testing_bearing_{j}', 'train.pt'))
        test = torch.load(os.path.join(data_path, f'Testing_bearing_{j}', 'test.pt'))
        train_data = train['samples']
        test_data = test['samples']
        max_ruls = test['max_ruls']
        print(f'Condition_{i}')
        print('Training', np.array(train_data).shape)
        print('Testing', np.array(test_data).shape)
        print(max_ruls)
    print('='*20)