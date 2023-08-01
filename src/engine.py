import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing import Optional, Tuple
import pandas as pd
from src.model import Model


class Engine:
    # TODO: Investigate the different loss functions and what this loss function does and when it should be used
    criterion = nn.BCEWithLogitsLoss()

    def __init__(self, model: Model, device: Optional[str]="cpu", optimizer: Optional[Optimizer] = None):
        self.device = device
        self.model = model.to(self.device)
        self.optimizer = optimizer


    def train(self, dataloader: DataLoader) -> float:
        self.model.train()
        final_loss = 0
        for inputs, targets in dataloader:
            self.optimizer.zero_grad()
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            loss = Engine.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            final_loss += loss.item()
        return final_loss / len(dataloader) # len(dataloader) is the number of batches

    """
        This function evaluates the model using the dataloader and the confusion matrix along with the dates.
    """
    def evaluate(self, dataloader: DataLoader, conf_matrix: dict) -> float:
        self.model.eval()
        final_loss = 0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            loss = Engine.criterion(outputs, targets)
            final_loss += loss.item()
            outputs_sigmoid = torch.sigmoid(outputs)
            outputs_binary = (outputs_sigmoid >= 0.5)
            conf_matrix['TP'] += ((targets == 1) & (outputs_binary == 1)).sum().item() # target o ground truth
            conf_matrix['FP'] += ((targets == 0) & (outputs_binary == 1)).sum().item()
            conf_matrix['TN'] += ((targets == 0) & (outputs_binary == 0)).sum().item()
            conf_matrix['FN'] += ((targets == 1) & (outputs_binary == 0)).sum().item()
        return final_loss / len(dataloader)
    

    def evaluate_with_LT(self, dataloader: DataLoader) -> pd.DataFrame:
        self.model.eval()
        test_df = pd.DataFrame()
        for LTs, inputs, targets in dataloader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            outputs_sigmoid = torch.sigmoid(outputs)
            outputs_binary = (outputs_sigmoid >= 0.5)
            tp = ((targets == 1) & (outputs_binary == 1)).cpu().numpy()
            fp = ((targets == 0) & (outputs_binary == 1)).cpu().numpy()
            tn = ((targets == 0) & (outputs_binary == 0)).cpu().numpy()
            fn = ((targets == 1) & (outputs_binary == 0)).cpu().numpy()

            for idx, LT in enumerate(LTs):
                LT = LT.detach().numpy().item()
                test_df = pd.concat([test_df, pd.DataFrame({'LT': LT, 'TP': tp[idx], 'FP': fp[idx], 'TN': tn[idx], 'FN': fn[idx]}, index=[0])], ignore_index=True)
        test_df['LT'] = pd.to_datetime(test_df.LT, unit='s')
        return test_df