import torch
from sklearn.base import BaseEstimator, ClassifierMixin

class Torch2DEstimator(BaseEstimator, ClassifierMixin):
    def __init__(self, model):
        self.model = model.eval()
        self.device = next(model.parameters()).device

    def fit(self, X, y=None):
        self.fitted_ = True
        return self  # nic nie robimy – model już wytrenowany

    def predict(self, X):
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            out = self.model.classifier(X_tensor)  # tylko klasyfikator
            return torch.argmax(out, dim=1).cpu().numpy()