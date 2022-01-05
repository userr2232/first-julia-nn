class Engine:
    def __init__(self, model, optimizer, device):
        self.device = device
        self.model = model.to(self.device)
        self.optimizer = optimizer

    @staticmethod
    def loss_fn(targets, outputs):
        pass

    def train(self, dataloader):
        self.model.train()
        final_loss = 0
        for inputs, targets in dataloader:
            self.optimizer.zero_grad()
            inputs = inputs.to(self.device)
            targets = inputs.to(self.device)
            outputs = self.model(inputs)
            loss = self.loss_fn(targets, outputs)
            loss.backward()
            self.optimizer.step()
            final_loss += loss.item()
        return final_loss / len(dataloader)

    def evaluate(self, dataloader):
        self.model.eval()
        final_loss = 0
        for inputs, targets in dataloader:
            inputs = inputs.to(self.device)
            targets = inputs.to(self.device)
            outputs = self.model(inputs)
            loss = self.loss_fn(targets, outputs)
            final_loss += loss.item()
        return final_loss / len(dataloader)