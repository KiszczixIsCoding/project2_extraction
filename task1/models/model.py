import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)    # pierwsza warstwa
        self.fc2 = nn.Linear(hidden_size, num_classes)   # druga warstwa

    def forward(self, x):
        x = self.fc1(x)          # warstwa 1
        x = F.relu(x)            # aktywacja (opcjonalna)
        x = self.fc2(x)          # warstwa 2 -> wynik (logity)
        return x

    def predict(self, x):
        self.eval()  # tryb ewaluacji (wyłącz dropout, batchnorm itd.)
        with torch.no_grad():  # nie liczymy gradientów (oszczędność pamięci)
            outputs = self.forward(x)
            _, preds = torch.max(outputs, dim=1)  # indeks klasy z najwyższym prawdopodobieństwem
        return preds