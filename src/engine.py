import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing import Optional, Tuple

from src.model import Model


class Engine:
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
        return final_loss / len(dataloader)

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
            conf_matrix['TP'] += ((targets == 1) & (outputs_binary == 1)).sum().item()
            conf_matrix['FP'] += ((targets == 0) & (outputs_binary == 1)).sum().item()
            conf_matrix['TN'] += ((targets == 0) & (outputs_binary == 0)).sum().item()
            conf_matrix['FN'] += ((targets == 1) & (outputs_binary == 0)).sum().item()
        return final_loss / len(dataloader)