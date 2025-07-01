import os

import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from captum.metrics import infidelity, sensitivity_max
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, ToPILImage
from torch import nn, optim
from skimage.segmentation import slic, mark_boundaries
from captum.attr import Lime, Saliency
import importer.import_datasets as importer
from task1.models.MnistDataset import MnistDataset
from task1.models.model import MLPModel
import task1.feature_extraction.features_mnist as feature_m
import params as p
import matplotlib.gridspec as gridspec

def perturb_fn(inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    noise = torch.randn_like(inputs) * 0.1
    perturbed = inputs + noise
    return perturbed, noise

def train_and_eval_model(model, train_loader, test_loader, num_epochs, draw=True):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    for epoch in range(num_epochs):
        model.train()
        batch_index = 0
        total_loss = 0

        df = pd.DataFrame({
            'predicted': [],
            'label': [],
            'features': []
        })

        print(f"Epoch {epoch}")
        for inputs, targets in train_loader:
            batch_index += 32
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}, Accuracy: {accuracy_score(df['predicted'], df['label'])}")

        with torch.no_grad():  # No gradients needed
            test_df = pd.DataFrame({
                'predicted': [],
                'label': [],
                'features': []
            })
            test_batch = 0
            for inputs, labels in test_loader:
                test_batch += 32
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)

                # New data to append (as DataFrame)
                new_data = pd.DataFrame({
                    'predicted': predicted,
                    'label': labels
                })

                # Append (concatenate)
                test_df = pd.concat([test_df, new_data], ignore_index=True)

        print(f'Accuracy1: {accuracy_score(test_df['predicted'], test_df['label'])}%')

def mnist_train():
    # collection of tuples in the form {Image image,int index}
    org_train_data = importer.import_mnist(True)
    org_test_data = importer.import_mnist(False)

    if p.feature_extract_mnist == 'verge_points':
        # comment lines below if you don't want to wait a couple of minutes (around 5 minutes) for preprocessing
        train_data = feature_m.verge_points(org_train_data)
        test_data = feature_m.verge_points(org_test_data)

    elif p.feature_extract_mnist == 'centroid_point':
        train_data = feature_m.centroid_point(org_train_data)
        test_data = feature_m.centroid_point(org_test_data)

    elif p.feature_extract_mnist == 'flatten_image':
        train_data = feature_m.flatten_image(org_train_data)
        test_data = feature_m.flatten_image(org_test_data)

    else:
        print("Feature extraction for dataset mnist in params.py is wrongly set. Check for typos.")

    trainDataset = MnistDataset(train_data[0], train_data[1])
    train_loader = DataLoader(trainDataset, batch_size=32, shuffle=True)

    testDataset = MnistDataset(test_data[0], test_data[1])
    test_loader = DataLoader(testDataset, batch_size=32, shuffle=False)

    mlpModel = MLPModel(len(train_data[0][0]), 512, 10)

    num_epochs = 25
    train_and_eval_model(mlpModel, train_loader, test_loader, num_epochs)

    for inputs, labels in test_loader:
        outputs = mlpModel(inputs)
        _, predicted = torch.max(outputs.data, 1)

        method = Saliency(mlpModel)
        for pred in predicted:
            attr = method.attribute(inputs, target=pred)
            infid = infidelity(mlpModel, perturb_func=perturb_fn, inputs=inputs, attributions=attr, target=pred)
            sensitivity = sensitivity_max(
                explanation_func=method.attribute,
                inputs=inputs,
                target=pred,
                n_perturb_samples=20,
                perturb_radius=0.01
            )
            print(infid)
            print(sensitivity)

    return mlpModel


model = mnist_train()



