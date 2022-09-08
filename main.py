import os.path
import sys
ROOT_DIR = os.path.abspath('')
sys.path.append('D:\good_memory\大三上\实训\MyPro')
sys.path.append(ROOT_DIR)
import torch
from util.data_processor import get_classification_dataset, get_ae_dataset
from util.train import fit_classification, fit_ae
from util.index import log
# from model.baseline import DL
from model.attention import Attention
from model.vae import VAE
from model.Myvae import BetaVAE
from torchsummary import summary
import argparse
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


parser = argparse.ArgumentParser(description='SDP main')
parser.add_argument('--model', type=str, required=True, help='choose a model')
parser.add_argument('--aft', type=str, required=False, default='simple', choices=['simple', 'full'], help='choose an aft structure')
args = parser.parse_args()

if torch.cuda.is_available():
    log("cuda available, using gpu to train and predict")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_path = 'dataset/csv/clean/scale.csv'

model_type = args.model
if model_type == 'dl' or model_type == 'attention':
    train_dataset, val_dataset, test_dataset, input_dim = get_classification_dataset(dataset_path, batch_size=128, val_rate=0.1, test_rate=0.1, over_sampling='smote', under_sampling=None, feature_num=None)
    if model_type == 'dl':
        model = DL(input_dim).to(device)
        log_path = './log/baseline.json'
        save_path = './model/saved_model/baseline.pt'
    elif model_type == 'attention':
        model = Attention(input_dim).to(device)
        log_path = './log/attention_%s.json' % args.aft
        save_path = './model/saved_model/attention_%s.pt' % args.aft
    else:
        model = None
        log_path = None
        save_path = None
        log('model %s not supported' % model_type)
        exit(1)
    summary(model.float(), (input_dim, ))

    multitask_options = {
        "alpha": 0.5,
        "gamma": 2.0,
        "average": True,
        "scale": 10.0
    }
    fit_classification(model=model.double(), train_dataset=train_dataset, epochs=1000, loss_function='fl', focal_loss_options=multitask_options, optimizer='adam', learning_rate=1e-4, val_dataset=val_dataset, test_dataset=test_dataset, log_path=log_path)
    torch.save(model.state_dict(), save_path)

elif model_type == 'vae':
    log_path = './log/vae.json'
    save_path = './model/saved_model/vae.pt'
    train_dataset, val_dataset_negative, val_dataset_positive, test_dataset_negative, test_dataset_positive, input_dim = get_ae_dataset(dataset_path, batch_size=64, val_rate=0.1, test_rate=0.1, feature_num=None)

    # print(len(train_dataset))
    # print(len(val_dataset_negative))
    # print(len(val_dataset_positive))
    # print(len(test_dataset_positive))
    # print(len(test_dataset_negative))
    # print(input_dim)
    #in_channels: int,
                 # latent_dim: int,
                 # hidden_dims: List = None,
                 # beta: int = 4,
                 # gamma: float = 1000.,
                 # max_capacity: int = 25,
                 # Capacity_max_iter: int = 1e5,
                 # loss_type: str = 'B',
                 # **kwargs) -> None:
    # 后期可以设置成一个超参数然后调节 latent_dim可调节，没有必要是40
    latent_dim = 80
    model = BetaVAE(40,latent_dim,0.001).to(device)
    # model = VAE(input_dim, abnormal_r0ate=0.1458938).to(device)

    # summary(model.float(), (input_dim,))
    fit_ae(model=model.double(), train_dataset=train_dataset, epochs=100, optimizer='adam', val_dataset=[val_dataset_negative, val_dataset_positive], test_dataset=[test_dataset_negative, test_dataset_positive], learning_rate=1e-4, early_stop=0.1, log_path=log_path)
    torch.save(model.state_dict(), save_path)

else:
    log('model %s not supported' % model_type)
    exit(1)
