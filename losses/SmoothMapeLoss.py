import torch

class SmoothMAPELoss(torch.nn.Module):
    def __init__(self, epsilon=1e-3):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        absolute_error = torch.abs(y_pred - y_true)
        relative_error = absolute_error / (torch.abs(y_true) + self.epsilon)
        return torch.mean(relative_error)