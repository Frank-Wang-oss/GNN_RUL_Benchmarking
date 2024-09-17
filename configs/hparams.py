
## The cuurent hyper-parameters values are not necessarily the best ones for a specific risk.
def get_hparams_class(dataset_name):
    """Return the algorithm class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


class CMAPSS():
    def __init__(self, dataset_id):
        super(CMAPSS, self).__init__()

        if dataset_id == 'FD001':
            self.train_params = {
                'FC_STGNN': {'num_epochs': 81,'batch_size': 100,'weight_decay': 1e-4,'learning_rate': 1e-3},
                'HierCorrPool': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                'LOGO': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-4, 'learning_rate': 1e-3, 'theta':0.001},
                'ASTGCNN': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                'STFA': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                'ST_Conv': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                'HAGCN': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-4, 'learning_rate': 1e-3, 'alpha':100},
                'RGCNU': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-4, 'learning_rate': 1e-3, 'lambda':0.1},
                'STAGNN': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                'DVGTformer': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                'GRU_CM': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                'STGNN': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-4, 'learning_rate': 1e-3},

            }

            self.alg_hparams = {
                'FC_STGNN': {'patch_size':25, 'num_patch':2, 'encoder_time_out':27, 'encoder_hidden_dim':8,
                             'encoder_out_dim':32, 'encoder_conv_kernel':2,'hidden_dim':8, 'num_sequential':6,
                             'num_node':14, 'num_windows':2},
                'HierCorrPool': {'patch_size': 25, 'num_patch': 2, 'input_dim': 10, 'hidden_dim': 10,
                                 'embedding_dim': 10, 'num_nodes': 14, 'encoder_conv_kernel': 8, 'num_nodes_out': 6},
                'LOGO': {'patch_size': 10, 'num_patch': 5, 'num_nodes': 14, 'hidden_dim': 8},
                'ASTGCNN': {'num_nodes':14, 'time_length':50, 'encoder_out_dim':50, 'output_dim':64, 'K':3},
                'STFA': {'patch_size':2, 'num_patch':25, 'num_nodes':14, 'hidden_dim':16, 'output_dim':5, 'encoder_hidden_dim':64, 'num_heads':10, 'dropout':0.2},
                'ST_Conv': {'num_nodes': 14, 'time_length': 50, 'kernel_size': 6},
                'HAGCN': {'patch_size':10, 'num_patch':5, 'hidden_dim':64, 'encoder_hidden_dim':60, 'output_dim':32},
                'RGCNU': {'num_nodes':14, 'time_length':50, 'hidden_dim':32, 'encoder_hidden_dim':32, 'kernel_size':3, 'alpha':1},
                'STAGNN': {'num_nodes': 14, 'time_length': 50, 'hidden_dim': 64, 'output_dim':10, 'num_heads':3, 'threshold':0},
                'DVGTformer': {'num_nodes': 14, 'time_length': 50, 'd_model': [144,248], 'num_heads': 4, 'lambda_param': 0.5, 'd_ff': [72,124],
                           'dropout': 0.1, 'num_blocks':3},
                'GRU_CM': {'num_nodes': 14, 'time_length': 50, 'gru_hidden_dim': 64},
                'STGNN': {'patch_size':50, 'num_patch':1, 'num_nodes': 14, 'hidden_dim':64, 'K':3, 'top_k': 10},

            }

        elif dataset_id == 'FD002':
            self.train_params = {
                'FC_STGNN': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                'HierCorrPool': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                'LOGO': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-4, 'learning_rate': 1e-3,
                         'theta': 0.01},
                'ASTGCNN': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                'STFA': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                'ST_Conv': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                'HAGCN': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-4, 'learning_rate': 1e-3, 'alpha':100},
                'RGCNU': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-4, 'learning_rate': 1e-3, 'lambda':0.1},
                'STAGNN': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                'DVGTformer': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                'GRU_CM': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                'STGNN': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-4, 'learning_rate': 1e-3},

            }
            self.alg_hparams = {
                'FC_STGNN': {'patch_size':1, 'num_patch':50, 'encoder_time_out':3, 'encoder_hidden_dim':8,
                             'encoder_out_dim':12, 'encoder_conv_kernel':2,'hidden_dim':8, 'num_sequential':10,
                             'num_node':14, 'num_windows':74},
                'HierCorrPool': {'patch_size': 10, 'num_patch': 5, 'input_dim': 10, 'hidden_dim': 10,
                                 'embedding_dim': 10, 'num_nodes': 14, 'encoder_conv_kernel': 12, 'num_nodes_out': 6},
                'LOGO': {'patch_size': 2, 'num_patch': 25, 'num_nodes': 14, 'hidden_dim': 6},
                'ASTGCNN': {'num_nodes': 14, 'time_length': 50, 'encoder_out_dim': 50, 'output_dim': 64, 'K': 3},
                'STFA': {'patch_size': 2, 'num_patch': 25, 'num_nodes': 14, 'hidden_dim': 16, 'output_dim': 5,
                         'encoder_hidden_dim': 64, 'num_heads': 10, 'dropout': 0.2},
                'ST_Conv': {'num_nodes': 14, 'time_length': 50, 'kernel_size': 6},
                'HAGCN': {'patch_size':25, 'num_patch':2, 'hidden_dim':64, 'encoder_hidden_dim':60, 'output_dim':32},
                'RGCNU': {'num_nodes': 14, 'time_length': 50, 'hidden_dim': 32, 'encoder_hidden_dim': 32,
                          'kernel_size': 3, 'alpha': 1},
                'STAGNN': {'num_nodes': 14, 'time_length': 50, 'hidden_dim': 16, 'output_dim': 10, 'num_heads': 3,
                           'threshold': 0},
                'DVGTformer': {'num_nodes': 14, 'time_length': 50, 'd_model': [144, 248], 'num_heads': 4,
                               'lambda_param': 0.5, 'd_ff': [72, 124],
                               'dropout': 0.1, 'num_blocks': 3},
                'GRU_CM': {'num_nodes': 14, 'time_length': 50, 'gru_hidden_dim': 64},
                'STGNN': {'patch_size': 50, 'num_patch': 1, 'num_nodes': 14, 'hidden_dim': 64, 'K': 3, 'top_k': 10},

            }
        elif dataset_id == 'FD003':
            self.train_params = {
                'FC_STGNN': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                'HierCorrPool': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                'LOGO': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-4, 'learning_rate': 1e-3,
                         'theta': 0.01},
                'ASTGCNN': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                'STFA': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                'ST_Conv': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                'HAGCN': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-4, 'learning_rate': 1e-3, 'alpha':100},
                'RGCNU': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-4, 'learning_rate': 1e-3, 'lambda':0.1},
                'STAGNN': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                'DVGTformer': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                'GRU_CM': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                'STGNN': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-4, 'learning_rate': 1e-3},

            }
            self.alg_hparams = {
                'FC_STGNN': {'patch_size':1, 'num_patch':50, 'encoder_time_out':3, 'encoder_hidden_dim':8,
                             'encoder_out_dim':6, 'encoder_conv_kernel':2,'hidden_dim':24, 'num_sequential':25,
                             'num_node':14, 'num_windows':74},
                'HierCorrPool': {'patch_size': 5, 'num_patch': 10, 'input_dim': 10, 'hidden_dim': 10,
                                 'embedding_dim': 10, 'num_nodes': 14, 'encoder_conv_kernel': 12, 'num_nodes_out': 6},
                'LOGO': {'patch_size': 10, 'num_patch': 5, 'num_nodes': 14, 'hidden_dim': 32},
                'ASTGCNN': {'num_nodes': 14, 'time_length': 50, 'encoder_out_dim': 50, 'output_dim': 64, 'K': 3},
                'STFA': {'patch_size': 2, 'num_patch': 25, 'num_nodes': 14, 'hidden_dim': 16, 'output_dim': 5,
                         'encoder_hidden_dim': 64, 'num_heads': 10, 'dropout': 0.2},
                'ST_Conv': {'num_nodes': 14, 'time_length': 50, 'kernel_size': 6},
                'HAGCN': {'patch_size':25, 'num_patch':2, 'hidden_dim':64, 'encoder_hidden_dim':60, 'output_dim':32},
                'RGCNU': {'num_nodes': 14, 'time_length': 50, 'hidden_dim': 32, 'encoder_hidden_dim': 32,
                          'kernel_size': 3, 'alpha': 1},
                'STAGNN': {'num_nodes': 14, 'time_length': 50, 'hidden_dim': 32, 'output_dim': 10, 'num_heads': 3,
                           'threshold': 0},
                'DVGTformer': {'num_nodes': 14, 'time_length': 50, 'd_model': [144, 248], 'num_heads': 4,
                               'lambda_param': 0.5, 'd_ff': [72, 124],
                               'dropout': 0.1, 'num_blocks': 3},
                'GRU_CM': {'num_nodes': 14, 'time_length': 50, 'gru_hidden_dim': 64},
                'STGNN': {'patch_size': 50, 'num_patch': 1, 'num_nodes': 14, 'hidden_dim': 64, 'K': 3, 'top_k': 10},

            }
        elif dataset_id == 'FD004':
            self.train_params = {
                'FC_STGNN': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                'HierCorrPool': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                'LOGO': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-4, 'learning_rate': 1e-3,
                         'theta': 0.001},
                'ASTGCNN': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                'STFA': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                'ST_Conv': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                'HAGCN': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-4, 'learning_rate': 1e-3, 'alpha':100},
                'RGCNU': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-4, 'learning_rate': 1e-3, 'lambda':0.1},
                'STAGNN': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                'DVGTformer': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                'GRU_CM': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                'STGNN': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-4, 'learning_rate': 1e-3},

            }
            self.alg_hparams = {
                'FC_STGNN': {'patch_size':2, 'num_patch':25, 'encoder_time_out':4, 'encoder_hidden_dim':8,
                             'encoder_out_dim':6, 'encoder_conv_kernel':2,'hidden_dim':8, 'num_sequential':10,
                             'num_node':14, 'num_windows':36},
                'HierCorrPool': {'patch_size': 10, 'num_patch': 5, 'input_dim': 10, 'hidden_dim': 10,
                                 'embedding_dim': 10, 'num_nodes': 14, 'encoder_conv_kernel': 12, 'num_nodes_out': 6},
                'LOGO': {'patch_size': 10, 'num_patch': 5, 'num_nodes': 14, 'hidden_dim': 10},
                'ASTGCNN': {'num_nodes': 14, 'time_length': 50, 'encoder_out_dim': 50, 'output_dim': 64, 'K': 3},
                'STFA': {'patch_size': 2, 'num_patch': 25, 'num_nodes': 14, 'hidden_dim': 16, 'output_dim': 5,
                         'encoder_hidden_dim': 64, 'num_heads': 10, 'dropout': 0.2},
                'ST_Conv': {'num_nodes': 14, 'time_length': 50, 'kernel_size': 6},
                'HAGCN': {'patch_size':50, 'num_patch':1, 'hidden_dim':64, 'encoder_hidden_dim':60, 'output_dim':32},
                'RGCNU': {'num_nodes': 14, 'time_length': 50, 'hidden_dim': 32, 'encoder_hidden_dim': 32,
                          'kernel_size': 3, 'alpha': 1},
                'STAGNN': {'num_nodes': 14, 'time_length': 50, 'hidden_dim': 32, 'output_dim': 10, 'num_heads': 3,
                           'threshold': 0},
                'DVGTformer': {'num_nodes': 14, 'time_length': 50, 'd_model': [144, 248], 'num_heads': 4,
                               'lambda_param': 0.5, 'd_ff': [72, 124],
                               'dropout': 0.1, 'num_blocks': 3},
                'GRU_CM': {'num_nodes': 14, 'time_length': 50, 'gru_hidden_dim': 64},
                'STGNN': {'patch_size': 50, 'num_patch': 1, 'num_nodes': 14, 'hidden_dim': 64, 'K': 3, 'top_k': 10},

            }
        else:
            raise ValueError('No input dataset id for CMAPSS')



class NCMAPSS():
    def __init__(self, dataset_id = None):
        super(NCMAPSS, self).__init__()

        self.train_params = {
            'FC_STGNN': {'num_epochs': 81,'batch_size': 100,'weight_decay': 1e-4,'learning_rate': 1e-3},
            'HierCorrPool': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
            'LOGO': {'num_epochs': 81, 'batch_size': 50, 'weight_decay': 0, 'learning_rate': 1e-3, 'theta':0.001},
            'ASTGCNN': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
            'ST_Conv': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
            'HAGCN': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-4, 'learning_rate': 1e-3, 'alpha':100},
            'RGCNU': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-4, 'learning_rate': 1e-3, 'lambda':0.1},
            'STAGNN': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
            'DVGTformer': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
            'GRU_CM': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
            'STGNN': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-4, 'learning_rate': 1e-3},

        }

        self.alg_hparams = {
            'FC_STGNN': {'patch_size':2, 'num_patch':25, 'encoder_time_out':4, 'encoder_hidden_dim':8,
                         'encoder_out_dim':32, 'encoder_conv_kernel':2,'hidden_dim':8, 'num_sequential':6,
                         'num_node':20, 'num_windows':36},
            'HierCorrPool': {'patch_size': 1, 'num_patch': 50, 'input_dim': 10, 'hidden_dim': 10,
                             'embedding_dim': 10, 'num_nodes': 20, 'encoder_conv_kernel': 32, 'num_nodes_out': 6},
            'LOGO': {'patch_size': 5, 'num_patch': 10, 'num_nodes': 20, 'hidden_dim': 10},
            'ASTGCNN': {'num_nodes':20, 'time_length':50, 'encoder_out_dim':50, 'output_dim':64, 'K':3},
            'ST_Conv': {'num_nodes': 20, 'time_length': 50, 'kernel_size': 6},
            'HAGCN': {'patch_size':25, 'num_patch':2, 'hidden_dim':64, 'encoder_hidden_dim':60, 'output_dim':32},
            'RGCNU': {'num_nodes':20, 'time_length':50, 'hidden_dim':32, 'encoder_hidden_dim':32, 'kernel_size':3, 'alpha':1},
            'STAGNN': {'num_nodes': 20, 'time_length': 50, 'hidden_dim': 32, 'output_dim': 10, 'num_heads': 3,
                           'threshold': 0},
            'DVGTformer': {'num_nodes': 20, 'time_length': 50, 'd_model': [144,248], 'num_heads': 4, 'lambda_param': 0.5, 'd_ff': [72,124],
                       'dropout': 0.1, 'num_blocks':3},
            'GRU_CM': {'num_nodes': 20, 'time_length': 50, 'gru_hidden_dim': 64},
            'STGNN': {'patch_size':10, 'num_patch':5, 'num_nodes': 20, 'hidden_dim':64, 'K':3, 'top_k': 10},

        }

class PHM2012():
    def __init__(self, condition_id):
        super(PHM2012, self).__init__()

        if condition_id == 'Condition_1':
            self.train_params = {
                'SAGCN': {'num_epochs': 81,'batch_size': 100,'weight_decay': 1e-4,'learning_rate': 1e-4},
                'STNet': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-2, 'learning_rate': 1e-2},
                'ST_GCN': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-4, 'learning_rate': 1e-4},
                'GAT_LSTM': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-4, 'learning_rate': 1e-4},
                'GDAGDL': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                'STMSGCN': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 0, 'learning_rate': 1e-2},
                'AGCN_TF': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-4, 'learning_rate': 1e-4},
                'LOGO_bearing': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-4, 'learning_rate': 1e-3,
                         'theta': 0.001},
                'HierCorrPool_bearing': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-4, 'learning_rate': 1e-3},

            }

            self.alg_hparams = {
                'SAGCN': {'num_patch':160, 'patch_size':16, 'gcn_hidden_dim':100, 'attention_hidden_dim':100},
                'STNet': {'num_patch':20, 'patch_size':128, 'num_nodes':9, 'nperseg':16, 'input_dim':9,
                          'Cheb_layers':[300,200,100], 'lstm_hidden_dim':10, 'autoencoder_hidden_dim':50},
                'ST_GCN': {'num_patch': 40, 'patch_size': 64, 'dropout': 0.2},
                'GAT_LSTM': {'num_patch': 40, 'patch_size': 64, 'hidden_dim':[300,200,100], 'lstm_hidden_dim':[30,20], 'dropout': 0.2},
                'GDAGDL': {'num_patch': 128, 'patch_size': 20, 'num_nodes': 3, 'nperseg': 4, 'input_dim': 6,
                          'gat_layer_dim': [300,150,50], 'lstm_hidden_dim': 20, 'autoencoder_hidden_dim': 256, 'autoencoder_out_dim':50},
                'STMSGCN': {'num_patch': 160, 'patch_size': 16, 'interval':6,'band_width':5, 'gcn_dims': [16, 64, 16, 1], 'gru_hidden_dim':8},
                'AGCN_TF': {'num_patch': 40, 'patch_size': 64, 'hidden_adj_dim': 100, 'hidden_gnn_dim': 100},
                'LOGO_bearing': {'patch_size': 64, 'num_patch': 40, 'input_dim': 9, 'num_nodes': 5, 'nperseg': 8, 'hidden_dim': 10},
                'HierCorrPool_bearing': {'patch_size': 32, 'num_patch': 80, 'input_dim': 5, 'hidden_dim': 10,
                                 'embedding_dim': 10, 'num_nodes': 5, 'nperseg': 8, 'encoder_conv_kernel': 48, 'num_nodes_out': 6},
            }

        elif condition_id == 'Condition_2':
            self.train_params = {
                'SAGCN': {'num_epochs': 81,'batch_size': 100,'weight_decay': 1e-4,'learning_rate': 1e-4},
                'STNet': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-2, 'learning_rate': 1e-2},
                'ST_GCN': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-4, 'learning_rate': 1e-4},
                'GAT_LSTM': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-4, 'learning_rate': 1e-4},
                'GDAGDL': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                'STMSGCN': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 0, 'learning_rate': 1e-2},
                'AGCN_TF': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-4, 'learning_rate': 1e-4},
                'LOGO_bearing': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-4, 'learning_rate': 1e-3,
                                 'theta': 0.001},
                'HierCorrPool_bearing': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-4,
                                         'learning_rate': 1e-3},

            }

            self.alg_hparams = {
                'SAGCN': {'num_patch':128, 'patch_size':20, 'gcn_hidden_dim':1000, 'attention_hidden_dim':200},
                'STNet': {'num_patch': 20, 'patch_size': 128, 'num_nodes': 9, 'nperseg': 16, 'input_dim': 9,
                          'Cheb_layers': [300, 200, 100], 'lstm_hidden_dim': 10, 'autoencoder_hidden_dim': 50},
                'ST_GCN': {'num_patch': 160, 'patch_size': 16, 'dropout': 0.2},
                'GAT_LSTM': {'num_patch': 80, 'patch_size': 32, 'hidden_dim': [300, 200, 100],
                             'lstm_hidden_dim': [30, 20], 'dropout': 0.2},
                'GDAGDL': {'num_patch': 128, 'patch_size': 20, 'num_nodes': 3, 'nperseg': 4, 'input_dim': 6,
                           'gat_layer_dim': [300, 150, 50], 'lstm_hidden_dim': 20, 'autoencoder_hidden_dim': 256,
                           'autoencoder_out_dim': 50},
                'STMSGCN': {'num_patch': 128, 'patch_size': 20, 'interval': 2, 'band_width': 3,
                            'gcn_dims': [16, 64, 16, 1], 'gru_hidden_dim': 8},
                'AGCN_TF': {'num_patch': 40, 'patch_size': 64, 'hidden_adj_dim': 100, 'hidden_gnn_dim': 100},
                'LOGO_bearing': {'patch_size': 64, 'num_patch': 40, 'input_dim': 9, 'num_nodes': 5,
                                 'nperseg': 8, 'hidden_dim': 10},
                'HierCorrPool_bearing': {'patch_size': 128, 'num_patch': 20, 'input_dim': 9, 'hidden_dim': 10,
                                         'embedding_dim': 10, 'num_nodes': 9, 'nperseg': 16, 'encoder_conv_kernel': 20,
                                         'num_nodes_out': 6},

            }
        elif condition_id == 'Condition_3':
            self.train_params = {
                'SAGCN': {'num_epochs': 81,'batch_size': 100,'weight_decay': 1e-4,'learning_rate': 1e-4},
                'STNet': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-2, 'learning_rate': 1e-2},
                'ST_GCN': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-4, 'learning_rate': 1e-4},
                'GAT_LSTM': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-4, 'learning_rate': 1e-4},
                'GDAGDL': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                'STMSGCN': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 0, 'learning_rate': 1e-2},
                'AGCN_TF': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-4, 'learning_rate': 1e-4},
                'LOGO_bearing': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-4, 'learning_rate': 1e-3,
                                 'theta': 0.001},
                'HierCorrPool_bearing': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-4,
                                         'learning_rate': 1e-3},

            }

            self.alg_hparams = {
                'SAGCN': {'num_patch':128, 'patch_size':20, 'gcn_hidden_dim':1000, 'attention_hidden_dim':200},
                'STNet': {'num_patch': 80, 'patch_size': 32, 'num_nodes': 5, 'nperseg': 8, 'input_dim': 5,
                          'Cheb_layers': [300, 200, 100], 'lstm_hidden_dim': 10, 'autoencoder_hidden_dim': 50},
                'ST_GCN': {'num_patch': 40, 'patch_size': 64, 'dropout': 0.2},
                'GAT_LSTM': {'num_patch': 40, 'patch_size': 64, 'hidden_dim': [300, 200, 100],
                             'lstm_hidden_dim': [30, 20], 'dropout': 0.2},
                'GDAGDL': {'num_patch': 80, 'patch_size': 32, 'num_nodes': 5, 'nperseg': 8, 'input_dim': 5,
                           'gat_layer_dim': [300, 150, 50], 'lstm_hidden_dim': 20, 'autoencoder_hidden_dim': 256,
                           'autoencoder_out_dim': 50},
                'STMSGCN': {'num_patch': 160, 'patch_size': 16, 'interval': 6, 'band_width': 5,
                            'gcn_dims': [16, 64, 16, 1], 'gru_hidden_dim': 8},
                'AGCN_TF': {'num_patch': 40, 'patch_size': 64, 'hidden_adj_dim': 100, 'hidden_gnn_dim': 100},
                'LOGO_bearing': {'patch_size': 64, 'num_patch': 40, 'input_dim': 9, 'num_nodes': 5,
                                 'nperseg': 8, 'hidden_dim': 10},
                'HierCorrPool_bearing': {'patch_size': 64, 'num_patch': 40, 'input_dim': 9, 'hidden_dim': 10,
                                         'embedding_dim': 10, 'num_nodes': 5, 'nperseg': 8, 'encoder_conv_kernel': 28,
                                         'num_nodes_out': 6},

            }
        else:
            raise ValueError('No input dataset id for PHM2012')



class XJTU_SY():
    def __init__(self, condition_id):
        super(XJTU_SY, self).__init__()

        if condition_id == 'Condition_1':
            self.train_params = {
                'SAGCN': {'num_epochs': 81,'batch_size': 100,'weight_decay': 1e-4,'learning_rate': 1e-4},
                'STNet': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-2, 'learning_rate': 1e-2},
                'ST_GCN': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-4, 'learning_rate': 1e-4},
                'GAT_LSTM': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-4, 'learning_rate': 1e-4},
                'GDAGDL': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                'STMSGCN': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 0, 'learning_rate': 1e-2},
                'AGCN_TF': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-4, 'learning_rate': 1e-4},
                'LOGO_bearing': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-4, 'learning_rate': 1e-3,
                                 'theta': 0.001},
                'HierCorrPool_bearing': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-4,
                                         'learning_rate': 1e-3},
            }

            self.alg_hparams = {
                'SAGCN': {'num_patch':32, 'patch_size':1024, 'gcn_hidden_dim':1000, 'attention_hidden_dim':100}, ### For SAGCN, adjust the value of patch_size
                'STNet': {'num_patch': 128, 'patch_size': 256, 'num_nodes': 9, 'nperseg': 16, 'input_dim': 17,
                          'Cheb_layers': [300, 200, 100], 'lstm_hidden_dim': 10, 'autoencoder_hidden_dim': 50},
                'ST_GCN': {'num_patch': 1024, 'patch_size': 32, 'dropout': 0.3},
                'GAT_LSTM': {'num_patch': 32, 'patch_size': 1024, 'hidden_dim': [300, 200, 100],
                             'lstm_hidden_dim': [30, 20], 'dropout': 0.2},
                'GDAGDL': {'num_patch': 32, 'patch_size': 1024, 'num_nodes': 17, 'nperseg': 32, 'input_dim': 33,
                           'gat_layer_dim': [300, 150, 50], 'lstm_hidden_dim': 20, 'autoencoder_hidden_dim': 256,
                           'autoencoder_out_dim': 50},
                'STMSGCN': {'num_patch': 256, 'patch_size': 128, 'interval': 3, 'band_width': 5,
                            'gcn_dims': [16, 64, 16, 1], 'gru_hidden_dim': 8},
                'AGCN_TF': {'num_patch': 128, 'patch_size': 256, 'hidden_adj_dim': 100, 'hidden_gnn_dim': 100},
                'LOGO_bearing': {'patch_size': 1024, 'num_patch': 32, 'input_dim': 33, 'num_nodes': 17,
                                 'nperseg': 32, 'hidden_dim': 10},
                'HierCorrPool_bearing': {'patch_size': 512, 'num_patch': 64, 'input_dim': 17, 'hidden_dim': 10,
                                         'embedding_dim': 10, 'num_nodes': 17, 'nperseg': 32, 'encoder_conv_kernel': 40,
                                         'num_nodes_out': 6},
            }

        elif condition_id == 'Condition_2':
            self.train_params = {
                'SAGCN': {'num_epochs': 81,'batch_size': 100,'weight_decay': 1e-4,'learning_rate': 1e-4},
                'STNet': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-2, 'learning_rate': 1e-2},
                'ST_GCN': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-4, 'learning_rate': 1e-4},
                'GAT_LSTM': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-4, 'learning_rate': 1e-4},
                'GDAGDL': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                'STMSGCN': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 0, 'learning_rate': 1e-2},
                'AGCN_TF': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-4, 'learning_rate': 1e-4},
                'LOGO_bearing': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-4, 'learning_rate': 1e-3,
                                 'theta': 0.001},
                'HierCorrPool_bearing': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-4,
                                         'learning_rate': 1e-3},
            }

            self.alg_hparams = {
                'SAGCN': {'num_patch':32, 'patch_size':1024, 'gcn_hidden_dim':1000, 'attention_hidden_dim':200},
                'STNet': {'num_patch': 32, 'patch_size': 1024, 'num_nodes': 17, 'nperseg': 32, 'input_dim': 33,
                          'Cheb_layers': [300, 200, 100], 'lstm_hidden_dim': 10, 'autoencoder_hidden_dim': 50},
                'ST_GCN': {'num_patch': 2048, 'patch_size': 16, 'dropout': 0.2},
                'GAT_LSTM': {'num_patch': 64, 'patch_size': 512, 'hidden_dim': [300, 200, 100],
                             'lstm_hidden_dim': [30, 20], 'dropout': 0.2},
                'GDAGDL': {'num_patch': 32, 'patch_size': 1024, 'num_nodes': 17, 'nperseg': 32, 'input_dim': 33,
                           'gat_layer_dim': [300, 150, 50], 'lstm_hidden_dim': 20, 'autoencoder_hidden_dim': 256,
                           'autoencoder_out_dim': 50},
                'STMSGCN': {'num_patch': 128, 'patch_size': 256, 'interval': 6, 'band_width': 10,
                            'gcn_dims': [16, 64, 16, 1], 'gru_hidden_dim': 8},
                'AGCN_TF': {'num_patch': 128, 'patch_size': 256, 'hidden_adj_dim': 100, 'hidden_gnn_dim': 100},
                'LOGO_bearing': {'patch_size': 1024, 'num_patch': 32, 'input_dim': 33, 'num_nodes': 17,
                                 'nperseg': 32, 'hidden_dim': 10},
                'HierCorrPool_bearing': {'patch_size': 256, 'num_patch': 128, 'input_dim': 17, 'hidden_dim': 10,
                                         'embedding_dim': 10, 'num_nodes': 9, 'nperseg': 16, 'encoder_conv_kernel': 72,
                                         'num_nodes_out': 6},
            }
        elif condition_id == 'Condition_3':
            self.train_params = {
                'SAGCN': {'num_epochs': 81,'batch_size': 100,'weight_decay': 1e-4,'learning_rate': 1e-4},
                'STNet': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-2, 'learning_rate': 1e-2},
                'ST_GCN': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-4, 'learning_rate': 1e-4},
                'GAT_LSTM': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-4, 'learning_rate': 1e-4},
                'GDAGDL': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-4, 'learning_rate': 1e-3},
                'STMSGCN': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 0, 'learning_rate': 1e-2},
                'AGCN_TF': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-4, 'learning_rate': 1e-4},
                'LOGO_bearing': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-4, 'learning_rate': 1e-3,
                                 'theta': 0.001},
                'HierCorrPool_bearing': {'num_epochs': 81, 'batch_size': 100, 'weight_decay': 1e-4,
                                         'learning_rate': 1e-3},
            }

            self.alg_hparams = {
                'SAGCN': {'num_patch':32, 'patch_size':1024, 'gcn_hidden_dim':1000, 'attention_hidden_dim':200},
                'STNet': {'num_patch': 64, 'patch_size': 512, 'num_nodes': 17, 'nperseg': 32, 'input_dim': 17,
                          'Cheb_layers': [300, 200, 100], 'lstm_hidden_dim': 10, 'autoencoder_hidden_dim': 50},
                'ST_GCN': {'num_patch': 2048, 'patch_size': 16, 'dropout': 0.2},
                'GAT_LSTM': {'num_patch': 32, 'patch_size': 1024, 'hidden_dim': [300, 200, 100],
                             'lstm_hidden_dim': [30, 20], 'dropout': 0.2},
                'GDAGDL': {'num_patch': 32, 'patch_size': 1024, 'num_nodes': 17, 'nperseg': 32, 'input_dim': 33,
                           'gat_layer_dim': [300, 150, 50], 'lstm_hidden_dim': 20, 'autoencoder_hidden_dim': 256,
                           'autoencoder_out_dim': 50},
                'STMSGCN': {'num_patch': 256, 'patch_size': 128, 'interval': 3, 'band_width': 5,
                            'gcn_dims': [16, 64, 16, 1], 'gru_hidden_dim': 8},
                'AGCN_TF': {'num_patch': 256, 'patch_size': 128, 'hidden_adj_dim': 100, 'hidden_gnn_dim': 100},
                'LOGO_bearing': {'patch_size': 1024, 'num_patch': 32, 'input_dim': 33, 'num_nodes': 17,
                                 'nperseg': 32, 'hidden_dim': 10},
                'HierCorrPool_bearing': {'patch_size': 256, 'num_patch': 128, 'input_dim': 17, 'hidden_dim': 10,
                                         'embedding_dim': 10, 'num_nodes': 9, 'nperseg': 16, 'encoder_conv_kernel': 72,
                                         'num_nodes_out': 6},
            }
        else:
            raise ValueError('No input dataset id for XJTU_SY')