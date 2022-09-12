from model.attention import Attention
from model.vae import VAE, VAELoss
from model.myxgboost import XGBoost
import torch
import pickle
import pandas as pd
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CDSDP:

    def __init__(self, input_dim):
        TW_attention = self.weight(0.94941, 0.91705, 0.98770)
        TW_XGBoost = self.weight(0.983984, 0.987844, 0.979116)
        TW_VAE = self.weight(0.86129, 0.52472, 0.52289)
        self.TW_attention = TW_attention / (TW_attention + TW_XGBoost + TW_VAE)
        self.TW_XGBoost = TW_XGBoost / (TW_attention + TW_XGBoost + TW_VAE)
        self.TW_VAE = TW_VAE / (TW_attention + TW_XGBoost + TW_VAE)

        model_path = '../model/saved_model/%s'
        self.attention = Attention(input_dim=input_dim).double().to(device)
        self.attention.load_state_dict(torch.load(model_path % 'attention_simple.pt'))
        self.attention.eval()
        self.vae = VAE(input_dim=input_dim, abnormal_rate=0.5).double().to(device)
        self.vae.load_state_dict(torch.load(model_path % 'vae_clip.pt', map_location='cpu'))
        self.vae.eval()
        self.xgboost = XGBoost()
        with open(model_path % 'xgboost.pkl', 'rb') as f:
            self.xgboost = pickle.load(f)

    def test(self, dataset, x):
        def compute_correct(pred, true):
            size = len(pred)
            correct_total = 0
            for i in range(size):
                if true[i] == 1.0:
                    if pred[i] >= 0.5:
                        correct_total += 1
                elif true[i] == 0.0:
                    if pred[i] < 0.5:
                        correct_total += 1
            return correct_total, size

        y_pred, y_true = self.predict(dataset, x)
        correct, length = compute_correct(y_pred, y_true)
        print(correct / length)

    def predict(self, dataset, x):
        y_pred_attention, y_true = self.forward_test(self.attention, dataset)
        y_pred_vae, _ = self.forward_test_ae(self.vae, dataset)
        y_pred_xgboost = self.xgboost.predict(x)
        y_pred = y_pred_attention * self.TW_attention + y_pred_vae * self.TW_VAE + y_pred_xgboost * self.TW_XGBoost
        return y_pred, y_true

    @staticmethod
    def weight(acc, pre, rec):
        return acc * 0.5 + pre * 0.25 + rec * 0.25

    @staticmethod
    def forward_test(model, dataset):
        y_pred = []
        y_true = []
        with torch.no_grad():
            for _, (x, y) in enumerate(dataset):
                y_pred += model(x.to(device)).cpu()
                y_true += y
        return np.array(pd.DataFrame(y_pred, columns=['value'])['value']), y_true

    @staticmethod
    def forward_test_ae(model, dataset):
        y_true = []
        loss = []
        loss_func = VAELoss()
        with torch.no_grad():
            for _, (x, y) in enumerate(dataset):
                x_hat, norm_x, mean_x, lv_x = model(x.to(device))
                loss += loss_func(x_hat, norm_x, mean_x, lv_x, reduction=False).cpu()
                y_true += y

        df_loss = pd.DataFrame(loss, columns=['loss'])
        index = [i for i in range(len(loss))]
        df_index = pd.DataFrame(index, columns=['index'])
        df = pd.concat([df_loss, df_index], axis=1)
        df.sort_values(by='loss', inplace=True, ascending=False, ignore_index=True)
        positive_num = int(len(loss) * model.abnormal_rate)
        df['pred'] = -1
        df.loc[0:positive_num, 'pred'] = 0.6
        df.loc[positive_num:, 'pred'] = 0.4
        df.sort_values(by='index', inplace=True, ignore_index=True)
        y_pred = np.array(df['pred'])
        return y_pred, y_true
