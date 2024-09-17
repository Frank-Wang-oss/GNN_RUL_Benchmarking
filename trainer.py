import torch
import torch.nn.functional as F

import os
# import wandb
import pandas as pd
import numpy as np
from dataloader.dataloader import data_generator
from configs.data_model_configs import get_dataset_class
from configs.hparams import get_hparams_class

from utils import fix_randomness, starting_logs, save_checkpoint, _calc_metrics_aeroengine, _calc_metrics_bearing, _calc_metrics
import warnings

import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

import collections
from algorithms.algorithms import get_algorithm_class
from utils import AverageMeter

torch.backends.cudnn.benchmark = True  # to fasten TCN


class GNN_RUL_trainer(object):
    """
   This class contain the main training functions for our AdAtime
    """
    def __init__(self, args):
        self.GNN_method = args.GNN_method  # Selected  DA Method
        self.dataset = args.dataset  # Selected  Dataset
        self.dataset_id = args.dataset_id
        self.device = torch.device(args.device)  # device
        self.bearing_id = args.bearing_id


        # Exp Description
        self.run_description = args.run_description
        self.experiment_description = args.experiment_description

        # paths
        if self.dataset == 'NCMAPSS':
            self.data_path = os.path.join(args.data_path, self.dataset)
        elif self.dataset == 'CMAPSS' or self.dataset == 'PHM2012':
            self.data_path = os.path.join(args.data_path, self.dataset, self.dataset_id)
        elif self.dataset == 'XJTU_SY':
            self.data_path = os.path.join(args.data_path, self.dataset, self.dataset_id, self.bearing_id)

        self.home_path = os.getcwd()
        self.save_dir = args.save_dir
        self.create_save_dir()

        # Specify runs
        self.num_runs = args.num_runs

        # get dataset and base model configs
        self.dataset_configs, self.hparams_class = self.get_configs(args.dataset_id)

        # Specify number of hparams
        self.train_configs = self.hparams_class.train_params[self.GNN_method]
        self.model_configs = self.hparams_class.alg_hparams[self.GNN_method]


        self.default_hparams = {**self.hparams_class.alg_hparams[self.GNN_method],
                                **self.hparams_class.train_params[self.GNN_method]}



    def train(self):
        run_name = f"{self.run_description}"
        # run = wandb.init(config=self.default_hparams, mode="online", name=run_name)
        #
        # self.hparams = wandb.config
        # Logging
        self.exp_log_dir = os.path.join(self.save_dir, self.experiment_description, run_name)
        os.makedirs(self.exp_log_dir, exist_ok=True)


        for run_id in range(self.num_runs):  # specify number of consecutive runs
            # fixing random seed
            fix_randomness(run_id)


            # Logging
            self.logger, self.log_dir = starting_logs(self.dataset, self.GNN_method, self.exp_log_dir,self.dataset_id, self.bearing_id, run_id)

            # Load data
            self.train_dl, self.test_dl, self.max_ruls = data_generator(self.data_path,  self.dataset_configs, self.train_configs)
            if isinstance(self.test_dl, dict):
                self.best_result = dict()
                for key in self.test_dl.keys():
                    self.best_result[key] = [[np.Inf], [np.Inf], [np.Inf], [np.Inf]]
            else:
                self.best_result = [[np.Inf], [np.Inf], [np.Inf], [np.Inf]]
            # get algorithm
            algorithm_class = get_algorithm_class(self.GNN_method)
            algorithm = algorithm_class(self.model_configs, self.train_configs, self.device)
            algorithm.to(self.device)

            # Average meters
            loss_avg_meters = collections.defaultdict(lambda: AverageMeter())

            # training..
            for epoch in range(1, self.train_configs["num_epochs"] + 1):
                algorithm.train()

                for step, (X, y) in enumerate(self.train_dl):
                    X, y = X.float().to(self.device), y.float().to(self.device)

                    losses = algorithm.update(X, y, epoch)

                    for key, val in losses.items():
                        loss_avg_meters[key].update(val, X.size(0))

                # logging
                self.logger.debug(f'[Epoch : {epoch}/{self.train_configs["num_epochs"]}]')
                for key, val in loss_avg_meters.items():
                    self.logger.debug(f'{key}\t: {val.avg:2.4f}')
                self.test_prediction(algorithm)

                self.calc_results_per_run(run_id)
                self.logger.debug(f'-------------------------------------')

            self.algorithm = algorithm
            save_checkpoint(self.home_path, self.algorithm, self.dataset_configs,
                            self.log_dir, self.default_hparams)





        # run.finish()

    def test_base(self, model, test_dataloader):
        pred_labels = np.array([])
        true_labels = np.array([])
        loss_total = []
        with torch.no_grad():
            for data, labels in test_dataloader:
                data = data.float().to(self.device)
                labels = labels.view((-1)).float().to(self.device)

                # forward pass
                predictions = model(data)
                predictions = predictions.view((-1))
                # compute loss
                loss = F.mse_loss(predictions, labels)
                loss_total.append(loss.item())
                pred = predictions.detach()  # get the index of the max log-probability

                pred_labels = np.append(pred_labels, pred.cpu().numpy())
                true_labels = np.append(true_labels, labels.cpu().numpy())
        return pred_labels, true_labels, loss_total
    def test_prediction(self,algorithm):
        model = algorithm.model.to(self.device)

        model.eval()

        if isinstance(self.test_dl, dict):
            test_pre = dict()
            test_real = dict()
            test_total_loss = dict()

            for key, test_dataloader_i in self.test_dl.items():
                pre_i, real_i, loss_i = self.test_base(model, test_dataloader_i)

                test_pre[key] = pre_i
                test_real[key] = real_i
                test_total_loss[key] = torch.tensor(loss_i).mean()

        else:
            test_pre, test_real, test_total_loss = self.test_base(model, self.test_dl)
            test_total_loss = torch.tensor(test_total_loss).mean()

        self.pred_labels = test_pre
        self.true_labels = test_real
        self.total_loss = test_total_loss




    def get_configs(self, dataset_id):
        dataset_class = get_dataset_class(self.dataset)
        hparams_class = get_hparams_class(self.dataset)

        dataset = dataset_class()
        hparams = hparams_class(dataset_id)

        return dataset, hparams


    def create_save_dir(self):
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

    def calc_results_per_run(self, run_id):

        indicator0 = 'Score_v1'
        indicator1 = 'Score_v2'
        indicator2 = 'MAE'
        indicator3 = 'RMSE'


        save_path = os.path.join(self.exp_log_dir, self.GNN_method + "_run_" + str(run_id))

        if isinstance(self.pred_labels, dict):
            keys = self.pred_labels.keys()

            for key in keys:

                ind0_i, ind1_i, ind2_i, ind3_i = _calc_metrics(self.pred_labels[key], self.true_labels[key],
                                                                self.max_ruls[key])

                if isinstance(key, float):
                    key_save = int(key)
                else:
                    key_save = key

                if ind3_i<self.best_result[key][3][-1]:
                    self.best_result[key][0].append(ind0_i)
                    self.best_result[key][1].append(ind1_i)
                    self.best_result[key][2].append(ind2_i)
                    self.best_result[key][3].append(ind3_i)

                    result_save_path = os.path.join(save_path, f"{key_save}_results.pt")
                    torch.save({'pre':self.pred_labels[key], 'real':self.true_labels[key], 'max_rul': self.max_ruls[key]}, result_save_path)

                df = pd.DataFrame({indicator0: self.best_result[key][0], indicator1: self.best_result[key][1],
                                   indicator2: self.best_result[key][2], indicator3: self.best_result[key][3]})
                scores_save_path = os.path.join(save_path, f"{key_save}_results.csv")
                df.to_csv(scores_save_path, index=False)
                self.logger.debug(f'Testing {key_save}, '
                                  f'{indicator0}: {self.best_result[key][0][-1]}, '
                                  f'{indicator1}: {self.best_result[key][1][-1]}, '
                                  f'{indicator2}: {self.best_result[key][2][-1]}, '
                                  f'{indicator3}: {self.best_result[key][3][-1]} '

                                  )


        else:
            ind0, ind1, ind2, ind3 = _calc_metrics(self.pred_labels, self.true_labels, self.max_ruls)


            if ind3 < self.best_result[3][-1]:
                self.best_result[0].append(ind0)
                self.best_result[1].append(ind1)
                self.best_result[2].append(ind2)
                self.best_result[3].append(ind3)
                result_save_path = os.path.join(save_path, f"results.pt")
                torch.save({'pre': self.pred_labels, 'real': self.true_labels, 'max_rul': self.max_ruls},
                           result_save_path)
            df = pd.DataFrame({indicator0: self.best_result[0], indicator1: self.best_result[1],
                               indicator2: self.best_result[2], indicator3: self.best_result[3]})
            scores_save_path = os.path.join(save_path, "results.csv")
            df.to_csv(scores_save_path, index=False)
            self.logger.debug(f'Testing, '
                              f'{indicator0}: {self.best_result[0][-1]}, '
                              f'{indicator1}: {self.best_result[1][-1]}, '
                              f'{indicator2}: {self.best_result[2][-1]}, '
                              f'{indicator3}: {self.best_result[3][-1]}'
                              )
