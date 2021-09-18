import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score
from torch import nn
from torch.nn import BCELoss, Linear
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau


class TimmModel(nn.Module):
    def __init__(self, model, sample_input, freeze):
        super(TimmModel, self).__init__()
        self.model = model
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        
        in_feature = model(sample_input).shape[1]
        self.linear1 = Linear(in_feature, in_feature // 4)
        self.linear2 = Linear(in_feature // 4, 1)

    def forward(self, wave, meta):
        x = self.model(wave)
        x = torch.cat([x, meta], dim=1)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = torch.sigmoid(x)
        return x


class Model(pl.LightningModule):
    def __init__(self, model, sample_input, optim, freeze):
        super(Model, self).__init__()
        self.model = TimmModel(model, sample_input, freeze)
        self.loss =  BCELoss()
        self.epsilon = 0
        self.optim = optim

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        wave, meta, y = batch
        y = torch.unsqueeze(y, dim=1)
        pred = self.model(wave, meta)
        _loss = self.loss(pred, y)
        self.logger.log_metrics({'loss': _loss})
        return {'loss': _loss}

    def validation_step(self, batch, batch_idx):
        wave, meta, y = batch
        pred = self.model(wave, meta)
        y_n = y.numpy()
        p_n = np.concatenate(pred.numpy())
        
        assert y_n.shape == p_n.shape

        val_auc = roc_auc_score(y_n, p_n)
        val_acc = accuracy_score(y_n, np.round(p_n))
        
        # print('===================')
        # print(np.round(p_n))
        # print('===================')
        # print(y_n)
        # print('===================')

        result = {'val_auc': val_auc, 'val_acc': val_acc}
        self.logger.log_metrics(result)
        return result

    def validation_end(self, outputs):
        auc = np.mean([out['val_auc'] for out in outputs])
        acc = np.mean([out['val_acc'] for out in outputs])
        return {'avg_val_auc':auc, 'avg_val_acc':acc}

    def configure_optimizers(self):
        self.optimizer = self.optim(self.parameters())
        return {
            'optimizer': self.optimizer,
            'lr_scheduler': CosineAnnealingLR(self.optimizer, T_max=64),
            'monitor': 'val_auc'
        }





