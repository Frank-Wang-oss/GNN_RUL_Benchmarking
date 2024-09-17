import os
import argparse
import warnings
from trainer import GNN_RUL_trainer
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

parser = argparse.ArgumentParser()


# ========  Experiments Name ================
parser.add_argument('--save_dir',               default='experiments_logs',                   type=str, help='Directory containing all experiments')
parser.add_argument('--experiment_description', default='Test_',                              type=str, help='Name of your experiment (CMAPSS, NCMAPSS, PHM2012, XJTU_SY')
parser.add_argument('--run_description',        default='test',                               type=str, help='name of your runs')

# ========= Select the GNN methods ============
## Aero-engine:
## Bearing:

parser.add_argument('--GNN_method',             default='FC_STGNN',                           type=str)

# ========= Select the DATASET ==============
parser.add_argument('--data_path',              default=r'./Data_Process/Processed_dataset',  type=str, help='Path containing dataset')
parser.add_argument('--dataset',                default='CMAPSS',                             type=str, help='Dataset choice: (CMAPSS - NCMAPSS - PHM2012 - XJTU_SY)')
parser.add_argument('--dataset_id',             default='FD004',                              type=str, help='Dataset ID choice: '
                                                                                                             '(CMAPSS: FD001. FD002, FD003, FD004'
                                                                                                             'NCMAPSS: None'
                                                                                                             'PHM2012: Condition_1,2,3'
                                                                                                             'XJTU_SY: Condition_1,2,3 - Testing_bearing_1,2,3,4,5)')
parser.add_argument('--bearing_id',             default='Testing_bearing_1',                  type=str, help='Bearing ID choice: Testing_bearing_1,2,3,4,5)')

# ========= Experiment settings ===============
parser.add_argument('--num_runs',               default=5,                                    type=int, help='Number of consecutive run with different seeds')
parser.add_argument('--device',                 default='cuda:0',                             type=str, help='cpu or cuda')




args = parser.parse_args()

if __name__ == "__main__":
    # GNN for aeroengine tasks
    method_aeroengine_names = ['ASTGCNN', 'GRU_CM', 'HAGCN', 'ST_Conv', 'STFA', 'RGCNU',
                    'STAGNN', 'HierCorrPool', 'LOGO', 'DVGTformer', 'STGNN', 'FC_STGNN']

    # GNN for bearing tasks
    method_bearing_names = ['ST_GCN', 'SAGCN', 'STNet', 'GAT_LSTM', 'STMSGCN', 'AGCN_TF', 'LOGO_bearing',
                    'HierCorrPool_bearing', 'GDAGDL']

    trainer = GNN_RUL_trainer(args)
    trainer.train()


