import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import optim, nn
from torch.utils.data import DataLoader

import importer.import_datasets as importer
import params as p
from analysis import get_all_metrics
from datasets.ImagenetteDataset import ImagenetteDataset
from datasets.MnistDataset import MnistDataset
from task1.plot_drawer import draw_accuracy, draw1, draw3
from task2.models.MNIST1_CNNModel import MNIST1_CNNModel
from task2.models.MNIST_CNNModel import MNIST_CNNModel

def train_and_eval_model(model, train_loader, test_loader, num_epochs, draw=True):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    preds = []
    accuracy_score_sets = {
        "train": [],
        "test": []
    }
    df = None
    metrics_scores = []
    # Define base path
    base_path = f"task2/results/{p.dataset2}2"

    # Get current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create full path
    folder_path = os.path.join(base_path, timestamp)

    # Make the directory
    os.makedirs(folder_path, exist_ok=True)

    for epoch in range(num_epochs):
        os.makedirs(f"{folder_path}/{epoch}", exist_ok=True)

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
            print(batch_index)
            print("Train")
            batch_index += 32
            outputs, features = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # New data to append (as DataFrame)
            new_data = pd.DataFrame({
                'predicted': predicted,
                'label': targets,
                'features': features.tolist()
            })
            # Append (concatenate)
            df = pd.concat([df, new_data], ignore_index=True)

        model.eval()
        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}, Accuracy: {accuracy_score(df['predicted'], df['label'])}")
        accuracy_score_sets["train"].append(accuracy_score(df['predicted'], df['label']))
        torch.save(model.state_dict(), f"{folder_path}/{epoch}/model_weights.pth")
        if draw:
            dff = pd.DataFrame(df['features'].tolist(), columns=['feature_1', 'feature_2'])
            dff = (dff - dff.min()) / (dff.max() - dff.min())
            dff_label = pd.DataFrame(df['label'].tolist(), columns=['label'])
            dff_final = pd.concat([dff, dff_label], ignore_index=True, axis=1)
            dff_final.columns = ['feature_1', 'feature_2', 'label']
            # draw3(dff_final, df['predicted'], f'{folder_path}/{epoch}/voronoi-train.png')
            draw3(dff_final, df['predicted'], model,f'{folder_path}/{epoch}/voronoi-train.png')

          # Set model to evaluation mode
        with torch.no_grad():  # No gradients needed
            test_df = pd.DataFrame({
                'predicted': [],
                'label': [],
                'features': []
            })

            test_batch = 0
            for inputs, labels in test_loader:
                test_batch += 32
                print(test_batch)
                outputs, features = model(inputs)
                _, predicted = torch.max(outputs.data, 1)

                # New data to append (as DataFrame)
                new_data = pd.DataFrame({
                    'predicted': predicted,
                    'label': labels,
                    'features': features.tolist()
                })

                # Append (concatenate)
                test_df = pd.concat([test_df, new_data], ignore_index=True)

            # cm = confusion_matrix(test_df['predicted'], test_df['label'])
            # disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            # disp.plot(cmap="Blues")
            # plt.show()
            preds = test_df['predicted']
            metrics_scores.append(get_all_metrics(test_df['label'], test_df['predicted']))
            if draw:
                test_dff = pd.DataFrame(test_df['features'].tolist(), columns=['feature_1', 'feature_2'])
                test_dff = (test_dff - test_dff.min()) / (test_dff.max() - test_dff.min())
                test_dff_label = pd.DataFrame(test_df['label'].tolist(), columns=['label'])
                test_dff_final = pd.concat([test_dff, test_dff_label], ignore_index=True, axis=1)
                test_dff_final.columns = ['feature_1', 'feature_2', 'label']
                # draw3(test_dff_final, test_df['predicted'], f'{folder_path}/{epoch}/voronoi-test.png')
                draw3(test_dff_final, test_df['predicted'], model,f'{folder_path}/{epoch}/voronoi-test.png')

        print(f'Accuracy1: {accuracy_score(test_df['predicted'], test_df['label'])}%')
        accuracy_score_sets["test"].append(accuracy_score(test_df['predicted'], test_df['label']))
        draw_accuracy(accuracy_score_sets, "wine", f"{folder_path}/{epoch}/accuracy", epoch + 1)

        cm_train = confusion_matrix(df['predicted'], df['label'])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_train)
        disp.plot(cmap="Blues")
        plt.savefig(f"{folder_path}/{epoch}/train_cm")
        plt.close()

        cm = confusion_matrix(test_df['predicted'], test_df['label'])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap="Blues")
        plt.savefig(f"{folder_path}/{epoch}/test_cm")
        plt.close()
    # cm_train = confusion_matrix(df['predicted'], df['label'])
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm_train)
    # disp.plot(cmap="Blues")
    # plt.show()
    #
    # cm = confusion_matrix(test_df['predicted'], test_df['label'])
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    # disp.plot(cmap="Blues")
    # plt.show()

    # draw_accuracy(accuracy_score_sets, "wine", None, num_epochs)
    return preds, metrics_scores

def main():
    org_train_data = None
    org_test_data = None

    if p.dataset2 == 'mnist':
        org_train_data = importer.import_mnist(True)
        org_test_data = importer.import_mnist(False)
    elif p.dataset2 == 'imagenette':
        org_train_data = importer.import_imagenette('train')
        org_test_data = importer.import_imagenette('val')

    train_loader = DataLoader(org_train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(org_test_data, batch_size=32, shuffle=False)
    mlpModel = MNIST1_CNNModel(32, 10)
    train_and_eval_model(mlpModel, train_loader, test_loader, 30)

main()