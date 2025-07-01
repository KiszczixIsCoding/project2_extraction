import torch
from torch import nn

class MNIST1_CNNModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MNIST1_CNNModel, self).__init__()
        self.features_extractor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=1),  # Ostatnia warstwa konwolucyjna z 64 kanałami
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 2, kernel_size=1),  # Ostatnia warstwa konwolucyjna z 64 kanałami
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # [B, 64, 1, 1]
            nn.Flatten()
        )

        with torch.no_grad():
            # dummy = torch.zeros(1, 3, 224, 224)
            dummy = torch.zeros(1, 1, 28, 28)
            flatten_dim = self.features_extractor(dummy).shape[1]

        self.classifier = nn.Sequential(
            nn.Linear(flatten_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.features_extractor(x)
        x = self.classifier(features)
        return x, features

    def predict(self, x):
        # tryb ewaluacji (wyłącz dropout, batchnorm itd.)
        with torch.no_grad():  # nie liczymy gradientów (oszczędność pamięci)
            outputs = self.classifier(x)
            _, preds = torch.max(outputs, 1)  # indeks klasy z najwyższym prawdopodobieństwem
        return preds