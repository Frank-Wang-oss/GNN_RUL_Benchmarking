import torch
import torch.nn as nn
import numpy as np

from models.FC_STGNN.Model import FC_STGNN_RUL
from models.HierCorrPool.Model import HierCorrPool_model
from models.LOGO.Model import LOGO_model
from models.ASTGCNN.Model import ASTGCNN_model
from models.STFA.Model import STFA_model
from models.ST_Conv.Model import ST_Conv_model
from models.HAGCN.Model import HAGCN_model
from models.RGCNU.Model import RGCNU_model
from models.STAGNN.Model import STAGNN_model
from models.DVGTformer.Model import DVGTformer_model
from models.GRU_CM.Model import GRU_CM_model
from models.STGNN.Model import STGNN_model
from models.SAGCN.Model import SAGCN_model
from models.STNet.Model import STNet_model
from models.ST_GCN.Model import ST_GCN_model
from models.GAT_LSTM.Model import GAT_LSTM_model
from models.GDAGDL.Model import GDAGDL_model
from models.STMSGCN.Model import STMSGCN_model
from models.AGCN_TF.Model import AGCN_TF_model
from models.LOGO_bearing.Model import LOGO_bearing_model
from models.HierCorrPool_bearing.Model import HierCorrPool_bearing_model



def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]


class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain adaptation algorithm.
    Subclasses should implement the update() method.
    """

    def __init__(self, configs):
        super(Algorithm, self).__init__()
        self.configs = configs
        self.mse = nn.MSELoss()

    def update(self, *args, **kwargs):
        raise NotImplementedError


class FC_STGNN(Algorithm):

    def __init__(self, configs, hparams, device):
        super(FC_STGNN, self).__init__(configs)

        # print(configs)

        self.model = FC_STGNN_RUL(**configs)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.hparams = hparams

    def update(self, X, y, epoch=None):
        predicted_RUL = self.model(X)

        loss = self.mse(predicted_RUL, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}


class HierCorrPool(Algorithm):

    def __init__(self, configs, hparams, device):
        super(HierCorrPool, self).__init__(configs)

        # print(configs)

        self.model = HierCorrPool_model(**configs)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.hparams = hparams

    def update(self, X, y, epoch=None):
        predicted_RUL = self.model(X)

        loss = self.mse(predicted_RUL, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}


class LOGO(Algorithm):

    def __init__(self, configs, hparams, device):
        super(LOGO, self).__init__(configs)

        # print(configs)

        self.model = LOGO_model(**configs)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.hparams = hparams
        self.theta = hparams["theta"]

        self.lr_schedular = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, [5, 10, 20, 25], 0.5)
    def update(self, X, y, epoch):
        predicted_RUL, loss_GL = self.model(X, GL = True)

        loss_mse = self.mse(predicted_RUL, y)

        loss = loss_mse+self.theta*loss_GL

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # self.lr_schedular.step()
        return {'loss': loss.item()}


class ASTGCNN(Algorithm):

    def __init__(self, configs, hparams, device):
        super(ASTGCNN, self).__init__(configs)

        # print(configs)
        self.model = ASTGCNN_model(**configs)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.hparams = hparams

    def update(self, X, y, epoch):
        predicted_RUL = self.model(X)

        loss_mse = self.mse(predicted_RUL, y)
        loss = loss_mse

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {'loss': loss.item()}


class STFA(Algorithm):

    def __init__(self, configs, hparams, device):
        super(STFA, self).__init__(configs)

        # print(configs)
        configs['device']= 'cuda:0'

        self.model = STFA_model(**configs)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.hparams = hparams

    def update(self, X, y, epoch):
        predicted_RUL = self.model(X)

        loss_mse = self.mse(predicted_RUL, y)
        loss = loss_mse

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {'loss': loss.item()}


class ST_Conv(Algorithm):

    def __init__(self, configs, hparams, device):
        super(ST_Conv, self).__init__(configs)

        # print(configs)

        self.model = ST_Conv_model(**configs)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.hparams = hparams

    def update(self, X, y, epoch):
        predicted_RUL = self.model(X)

        loss_mse = self.mse(predicted_RUL, y)
        loss = loss_mse

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {'loss': loss.item()}

class HAGCN(Algorithm):

    def __init__(self, configs, hparams, device):
        super(HAGCN, self).__init__(configs)

        # print(configs)

        self.model = HAGCN_model(**configs)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.hparams = hparams

        self.alpha = hparams["alpha"]
    def update(self, X, y, epoch):
        predicted_RUL, KL_Loss = self.model(X, train=True)

        loss_mse = self.mse(predicted_RUL, y)
        loss = loss_mse + self.alpha*KL_Loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {'loss': loss.item()}

class RGCNU(Algorithm):

    def __init__(self, configs, hparams, device):
        super(RGCNU, self).__init__(configs)

        # print(configs)

        self.model = RGCNU_model(**configs)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.hparams = hparams

        self.lambda_hy = hparams["lambda"]

    def mean_std_loss(self, pre, real, pre_std):
        ## pre size (bs, 1)
        # pre_std = pre_std+1e-10
        # s = torch.log(pre_std**2)

        square_diff = (pre-real)**2

        loss_mean = 1/2*torch.exp(-pre_std)*square_diff + 1/2*pre_std
        loss_mean = torch.mean(loss_mean)
        loss_std = torch.exp(2*pre_std)
        loss_std = torch.sum(loss_std)

        loss = (1-self.lambda_hy)*loss_mean + self.lambda_hy * loss_std

        return loss

    def update(self, X, y, epoch):
        predicted_RUL, std_RUL = self.model(X, train = True)

        loss_mse = self.mse(predicted_RUL, y)
        # loss_mse = self.mean_std_loss(predicted_RUL, std_RUL, y)

        loss = loss_mse

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {'loss': loss.item()}


class STAGNN(Algorithm):

    def __init__(self, configs, hparams, device):
        super(STAGNN, self).__init__(configs)

        # print(configs)

        self.model = STAGNN_model(**configs)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.hparams = hparams

    def update(self, X, y, epoch):
        predicted_RUL = self.model(X)

        loss_mse = self.mse(predicted_RUL, y)
        loss = loss_mse

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {'loss': loss.item()}


class DVGTformer(Algorithm):

    def __init__(self, configs, hparams, device):
        super(DVGTformer, self).__init__(configs)

        # print(configs)

        self.model = DVGTformer_model(**configs)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.hparams = hparams

    def update(self, X, y, epoch):
        predicted_RUL = self.model(X)

        loss_mse = self.mse(predicted_RUL, y)
        loss = loss_mse

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {'loss': loss.item()}



class GRU_CM(Algorithm):

    def __init__(self, configs, hparams, device):
        super(GRU_CM, self).__init__(configs)

        # print(configs)

        self.model = GRU_CM_model(**configs)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.hparams = hparams

    def update(self, X, y, epoch):
        predicted_RUL = self.model(X)

        loss_mse = self.mse(predicted_RUL, y)
        loss = loss_mse

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {'loss': loss.item()}


class STGNN(Algorithm):

    def __init__(self, configs, hparams, device):
        super(STGNN, self).__init__(configs)

        # print(configs)

        self.model = STGNN_model(**configs)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.hparams = hparams

    def update(self, X, y, epoch):
        predicted_RUL = self.model(X)

        loss_mse = self.mse(predicted_RUL, y)
        loss = loss_mse

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {'loss': loss.item()}

## Bearing Algo
class SAGCN(Algorithm):

    def __init__(self, configs, hparams, device):
        super(SAGCN, self).__init__(configs)

        # print(configs)

        self.model = SAGCN_model(**configs)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.hparams = hparams

    def update(self, X, y, epoch):
        predicted_RUL = self.model(X)

        loss_mse = self.mse(predicted_RUL, y)
        loss = loss_mse

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {'loss': loss.item()}

class STNet(Algorithm):

    def __init__(self, configs, hparams, device):
        super(STNet, self).__init__(configs)

        # print(configs)

        self.model = STNet_model(**configs)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.hparams = hparams

    def update(self, X, y, epoch):
        predicted_RUL, reconstruction_losss = self.model(X, train = True)

        loss_mse = self.mse(predicted_RUL, y)
        loss = loss_mse + reconstruction_losss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {'loss': loss.item()}

class ST_GCN(Algorithm):

    def __init__(self, configs, hparams, device):
        super(ST_GCN, self).__init__(configs)

        # print(configs)

        self.model = ST_GCN_model(**configs)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.hparams = hparams

    def update(self, X, y, epoch):
        predicted_RUL = self.model(X)

        loss_mse = self.mse(predicted_RUL, y)
        loss = loss_mse

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {'loss': loss.item()}

class GAT_LSTM(Algorithm):

    def __init__(self, configs, hparams, device):
        super(GAT_LSTM, self).__init__(configs)

        # print(configs)

        self.model = GAT_LSTM_model(**configs)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.hparams = hparams

    def update(self, X, y, epoch):
        predicted_RUL = self.model(X)

        loss_mse = self.mse(predicted_RUL, y)
        loss = loss_mse

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {'loss': loss.item()}

class GDAGDL(Algorithm):

    def __init__(self, configs, hparams, device):
        super(GDAGDL, self).__init__(configs)

        # print(configs)

        self.model = GDAGDL_model(**configs)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.hparams = hparams

    def update(self, X, y, epoch):
        predicted_RUL, reconstruction_losss = self.model(X, train = True)

        loss_mse = self.mse(predicted_RUL, y)
        loss = loss_mse + reconstruction_losss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {'loss': loss.item()}

class STMSGCN(Algorithm):

    def __init__(self, configs, hparams, device):
        super(STMSGCN, self).__init__(configs)

        # print(configs)

        self.model = STMSGCN_model(**configs)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.hparams = hparams

    def update(self, X, y, epoch=None):
        predicted_RUL = self.model(X)

        loss = self.mse(predicted_RUL, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}


class AGCN_TF(Algorithm):

    def __init__(self, configs, hparams, device):
        super(AGCN_TF, self).__init__(configs)

        # print(configs)

        self.model = AGCN_TF_model(**configs)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.hparams = hparams

    def update(self, X, y, epoch=None):
        predicted_RUL = self.model(X)

        loss = self.mse(predicted_RUL, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

class LOGO_bearing(Algorithm):

    def __init__(self, configs, hparams, device):
        super(LOGO_bearing, self).__init__(configs)

        # print(configs)

        self.model = LOGO_bearing_model(**configs)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.hparams = hparams
        self.theta = hparams["theta"]

        self.lr_schedular = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, [5, 10, 20, 25], 0.5)
    def update(self, X, y, epoch):
        predicted_RUL, loss_GL = self.model(X, GL = True)

        loss_mse = self.mse(predicted_RUL, y)
        loss = loss_mse+self.theta*loss_GL

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_schedular.step()
        return {'loss': loss.item()}



class HierCorrPool_bearing(Algorithm):

    def __init__(self, configs, hparams, device):
        super(HierCorrPool_bearing, self).__init__(configs)

        # print(configs)

        self.model = HierCorrPool_bearing_model(**configs)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.hparams = hparams

    def update(self, X, y, epoch=None):
        predicted_RUL = self.model(X)

        loss = self.mse(predicted_RUL, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}