from torch import nn


class Engine:
    criterion = nn.BCEWithLogitsLoss()

    def __init__(self, model, optimizer, device):
        self.device = device
        self.model = model.to(self.device)
        self.optimizer = optimizer

    def train(self, dataloader):
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

    def evaluate(self, dataloader):
        self.model.eval()
        final_loss = 0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            loss = Engine.criterion(outputs, targets)
            final_loss += loss.item()
        return final_loss / len(dataloader)