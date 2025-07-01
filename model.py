import os

import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage

import params as p
import importer.import_datasets as importer
import task1.feature_extraction.features_mnist as feature_m
from analysis import get_all_metrics
from task1.models.MnistDataset import MnistDataset
from task1.models.RealDataset import RealDataset
from task1.models.model import MLPModel
from sklearn.metrics import accuracy_score

from task1.plot_drawer import draw_accuracy, draw3, draw1, draw_linechart

org_train_data, org_test_data = None, None
train_data, test_data = None, None
prediction = None

def train_and_eval_model(model, train_loader, test_loader, num_epochs):
    model.train()
    preds = []
    accuracy_score_sets = {
        "train": [],
        "test": []
    }
    df = None
    metrics_scores = []
    for epoch in range(num_epochs):
        model.train()
        batch_index = 0
        total_loss = 0

        df = pd.DataFrame({
            'predicted': [],
            'label': []
        })
        for inputs, targets in train_loader:
            batch_index += 32
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # New data to append (as DataFrame)
            new_data = pd.DataFrame({
                'predicted': predicted,
                'label': targets
            })
            # Append (concatenate)
            df = pd.concat([df, new_data], ignore_index=True)

        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}, Accuracy: {accuracy_score(df['predicted'], df['label'])}")
        accuracy_score_sets["train"].append(accuracy_score(df['predicted'], df['label']))

        model.eval()  # Set model to evaluation mode
        with torch.no_grad():  # No gradients needed
            test_df = pd.DataFrame({
                'predicted': [],
                'label': []
            })

            for inputs, labels in test_loader:
                outputs = mlpModel(inputs)
                _, predicted = torch.max(outputs.data, 1)

                # New data to append (as DataFrame)
                new_data = pd.DataFrame({
                    'predicted': predicted,
                    'label': labels
                })

                # Append (concatenate)
                test_df = pd.concat([test_df, new_data], ignore_index=True)

            # cm = confusion_matrix(test_df['predicted'], test_df['label'])
            # disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            # disp.plot(cmap="Blues")
            # plt.show()
            preds = test_df['predicted']
            metrics_scores.append(get_all_metrics(test_df['label'], test_df['predicted']))

        print(f'Accuracy1: {accuracy_score(test_df['predicted'], test_df['label'])}%')
        accuracy_score_sets["test"].append(accuracy_score(test_df['predicted'], test_df['label']))


    cm_train = confusion_matrix(df['predicted'], df['label'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_train)
    disp.plot(cmap="Blues")
    plt.show()

    cm = confusion_matrix(test_df['predicted'], test_df['label'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.show()

    draw_accuracy(accuracy_score_sets, "wine", None, num_epochs)
    return preds, metrics_scores

if p.dataset == 'mnist':
    # collection of tuples in the form {Image image,int index}
    org_train_data = importer.import_mnist(True)
    org_test_data = importer.import_mnist(False)

    save_dir = "mnist_images"
    os.makedirs(save_dir, exist_ok=True)
    to_pil = ToPILImage()
    for i in range(10):
        image, label = org_train_data[i]
        image.save(os.path.join(save_dir, f"digit_{label}_{i}.png"))

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

    print(train_data[0][0])
    # mlpModel = MLPModel(len(train_data[0][0]), 512, 10)
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(mlpModel.parameters(), lr=0.1)
    # num_epochs = 100

    # mlpModel = MLPModel(len(train_data[0][0]), 512, 10)
    mlpModel = MLPModel(len(train_data[0][0]), 512, 10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(mlpModel.parameters(), lr=0.1)
    num_epochs = 100
    preds, metrics_scores = train_and_eval_model(mlpModel, train_loader, test_loader, num_epochs)
    mnist_df = pd.DataFrame({
        'feature_1': [f[0] for f in test_data[0]],
        'feature_2': [f[1] for f in test_data[0]],
        'label': test_data[1]
    })
    print("DRAW LINECHART")
    draw_linechart(metrics_scores, "mnist")
    draw1(mnist_df, preds)

else:
    data = None
    if p.dataset == 'breast_cancer':
        data = importer.load_breast_cancer()
    elif p.dataset == 'iris':
        data = importer.load_iris()
    elif p.dataset == 'wine':
        data = importer.load_wine()

    train_data, test_data, train_target, test_target = train_test_split(data['data'], data['target'], test_size=0.2, random_state=42)
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)

    trainRealDataset = RealDataset(train_data, train_target)
    testRealDataset = RealDataset(test_data, test_target)

    train_loader = DataLoader(trainRealDataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(testRealDataset, batch_size=32, shuffle=True)

    print(len(set(train_target)))
    mlpModel = MLPModel((len(data['data'].columns)), 512, len(set(train_target)))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(mlpModel.parameters(), lr=0.1)
    num_epochs = 100

    train_and_eval_model(mlpModel, train_loader, test_loader, num_epochs)


# test_loader is a DataLoader with your test_data

# if train_data is not None and test_data is not None:
#     ## TODO: Run training and prediction
#
#     if prediction is not None:
#         acc, prec, recall = count_base_metrics(test_data[1], prediction)
#         metrics_table, multilabel_confusion_matrix, classification_report = None, None, None
#
#         if p.metrics_extended is True:
#             metrics_table, multilabel_confusion_matrix, classification_report = count_extended_metrics(test_data[1],
#                                                                                                        prediction)
#
#         if p.save_for_analysis is True:
#             analysis_dict = create_analysis_directory()
#
#             if p.dataset == 'mnist' and p.save_img is True:
#                 save_wrong_classified_imgs(analysis_dict, prediction, test_data, org_test_data)
#
#             save_params(analysis_dict)
#             save_metrics(analysis_dict, acc, prec, recall)
#
#             if p.metrics_extended is True:
#                 save_extended_metrics(analysis_dict, acc, prec, recall, metrics_table, multilabel_confusion_matrix,
#                                       classification_report)
#             else:
#                 print_metrics(acc, prec, recall)
#
#                 if p.metrics_extended is True:
#                     print_extended_metrics(metrics_table, multilabel_confusion_matrix, classification_report)
