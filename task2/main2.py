import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
# from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
from torch import optim, nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# import importer.import_datasets as importer
import params as p
# from analysis import get_all_metrics
# from datasets.ImagenetteDataset import ImagenetteDataset
# from datasets.MnistDataset import MnistDataset
# from task1.plot_drawer import draw_accuracy, draw1, draw3
# from task2.models.MNIST1_CNNModel import MNIST1_CNNModel
# from task2.models.MNIST_CNNModel import MNIST_CNNModel

class MNIST_CNNModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MNIST_CNNModel, self).__init__()
        self.features_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # Ostatnia warstwa konwolucyjna z 64 kanałami
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1)),  # [B, 64, 1, 1]
            nn.Flatten()
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            # dummy = torch.zeros(1, 1, 28, 28)
            flatten_dim = self.features_extractor(dummy).shape[1]

        print(f"FLatten {flatten_dim}")
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
        self.eval()  # tryb ewaluacji (wyłącz dropout, batchnorm itd.)
        with torch.no_grad():  # nie liczymy gradientów (oszczędność pamięci)
            outputs = self.forward(x)
            _, preds = torch.max(outputs, dim=1)  # indeks klasy z najwyższym prawdopodobieństwem
        return preds

transform_translations_imagenette = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(p=1.0),  # Losowe odbicie w poziomie (50% szansa)
    transforms.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.2, hue=0.1),  # Zmiany koloru, kontrastu i nasycenia
    transforms.ToTensor(),  # konwertuje do [0, 1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])  # normalizacja jak w ImageNet
])

transform_mnist = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)) # zmienia PIL.Image na tensor [0,1]
])

transform_translation_mnist = transforms.Compose([
    # transforms.RandomRotation(degrees=15),  # Losowa rotacja do ±10°
    # transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),  # Przesunięcie w poziomie/pionie do 10%
    # transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.3),  # Szum (30% szansa)
    # transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),  # Zmiana ostrości (30% szansa)
    transforms.ToTensor(),  # Konwersja do tensora
    transforms.Normalize((0.1307,), (0.3081,))
])

def import_mnist(train):
    return datasets.MNIST(root='../resources', train=train, download=True, transform=transform_mnist)

def import_mnist_subset(train, subset_len):
    mnist_dataset = datasets.MNIST(root='../resources', train=train, download=True, transform=transform_translation_mnist)

    # for i in range(15):
    #     image_tensor, label = mnist_dataset[i]
    #
    #     # Odpinamy normalizację dla wyświetlenia (odwracamy standaryzację)
    #     unnormalized = image_tensor * 0.3081 + 0.1307
    #
    #     # Wyświetlamy
    #     plt.imshow(unnormalized.squeeze(), cmap='gray')
    #     plt.title(f"Label: {label}")
    #     plt.axis('off')
    #     plt.show()
    #
    # print(int(subset_len / 10))
    class_counts = {i: 0 for i in range(int(subset_len / 10))}
    selected_indices = []
    for idx, (_, label) in enumerate(mnist_dataset):
        label = int(label)
        if class_counts[label] < int(subset_len / 10):
            selected_indices.append(idx)
            class_counts[label] += 1
        if sum(class_counts.values()) >= subset_len:
            break

    # Stwórz ograniczony dataset
    balanced_subset = Subset(mnist_dataset, selected_indices)
    print("Rozmiar nowego zbioru:", len(balanced_subset))
    labels = [balanced_subset[i][1] for i in range(len(balanced_subset))]
    print("Rozkład klas:", {i: labels.count(i) for i in range(int(subset_len / 10))})
    print(type(balanced_subset))

    return balanced_subset

# def import_imagenette(train_str):
#     return datasets.Imagenette(root='../resources-imagenette',split=train_str, download=True, transform=transform1)

def unnormalize(tensor, mean, std):
    mean = torch.tensor(mean).view(3, 1, 1)  # Dopasowanie wymiarów
    std = torch.tensor(std).view(3, 1, 1)
    return tensor * std + mean

def import_imagenette_subset(train_str, subset_len):
    imagenette_dataset = datasets.Imagenette(root='../resources-imagenette',split=train_str, download=True, transform=transform_translations_imagenette)
    # imagenette_dataset1 = datasets.Imagenette(root='../resources-imagenette',split=train_str, download=True, transform=transform1)

    # for i in range(15):
    #     image_tensor, label = imagenette_dataset[8000 + i]
    #     image_tensor1, label1 = imagenette_dataset1[8000 + i]
    #
    #     # Odpinamy normalizację dla wyświetlenia (odwracamy standaryzację)
    #     unnormalized = unnormalize(image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #     unnormalized1 = unnormalize(image_tensor1, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #     unnormalized_image = unnormalized.permute(1, 2, 0).cpu().numpy()
    #     unnormalized_image1 = unnormalized1.permute(1, 2, 0).cpu().numpy()
    #     # Wyświetlamy
    #     plt.title(f"Label: {label1}")
    #     plt.imshow(unnormalized_image1)
    #     plt.show()
    #     plt.imshow(unnormalized_image)
    #     plt.title(f"Label: {label}")
    #     plt.axis('off')
    #     plt.show()

    print(int(subset_len / 10))
    class_counts = {i: 0 for i in range(int(subset_len / 10))}
    selected_indices = []
    for idx, (_, label) in enumerate(imagenette_dataset):
        label = int(label)
        if class_counts[label] < int(subset_len / 10):
            selected_indices.append(idx)
            class_counts[label] += 1
        if sum(class_counts.values()) >= subset_len:
            break

    # Stwórz ograniczony dataset
    balanced_subset = Subset(imagenette_dataset, selected_indices)
    print("Rozmiar nowego zbioru:", len(balanced_subset))
    labels = [balanced_subset[i][1] for i in range(len(balanced_subset))]
    print("Rozkład klas:", {i: labels.count(i) for i in range(int(subset_len / 10))})
    print(type(balanced_subset))

    return balanced_subset



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
    base_path = f"task2/results2/{p.dataset2}"

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
        # if draw:
        #     dff = pd.DataFrame(df['features'].tolist(), columns=['feature_1', 'feature_2'])
        #     dff = (dff - dff.min()) / (dff.max() - dff.min())
        #     dff_label = pd.DataFrame(df['label'].tolist(), columns=['label'])
        #     dff_final = pd.concat([dff, dff_label], ignore_index=True, axis=1)
        #     dff_final.columns = ['feature_1', 'feature_2', 'label']
        #     # draw3(dff_final, df['predicted'], f'{folder_path}/{epoch}/voronoi-train.png')
        #     draw3(dff_final, df['predicted'], model,f'{folder_path}/{epoch}/voronoi-train.png')

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
            # preds = test_df['predicted']
            # metrics_scores.append(get_all_metrics(test_df['label'], test_df['predicted']))
            # if draw:
            #     test_dff = pd.DataFrame(test_df['features'].tolist(), columns=['feature_1', 'feature_2'])
            #     test_dff = (test_dff - test_dff.min()) / (test_dff.max() - test_dff.min())
            #     test_dff_label = pd.DataFrame(test_df['label'].tolist(), columns=['label'])
            #     test_dff_final = pd.concat([test_dff, test_dff_label], ignore_index=True, axis=1)
            #     test_dff_final.columns = ['feature_1', 'feature_2', 'label']
            #     # draw3(test_dff_final, test_df['predicted'], f'{folder_path}/{epoch}/voronoi-test.png')
            #     draw3(test_dff_final, test_df['predicted'], model,f'{folder_path}/{epoch}/voronoi-test.png')

        print(f'Accuracy1: {accuracy_score(test_df["predicted"], test_df["label"])}%')
        accuracy_score_sets["test"].append(accuracy_score(test_df['predicted'], test_df['label']))
        # draw_accuracy(accuracy_score_sets, "wine", f"{folder_path}/{epoch}/accuracy", epoch + 1)

        # cm_train = confusion_matrix(df['predicted'], df['label'])
        # disp = ConfusionMatrixDisplay(confusion_matrix=cm_train)
        # disp.plot(cmap="Blues")
        # plt.savefig(f"{folder_path}/{epoch}/train_cm")
        # plt.close()
        #
        # cm = confusion_matrix(test_df['predicted'], test_df['label'])
        # disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        # disp.plot(cmap="Blues")
        # plt.savefig(f"{folder_path}/{epoch}/test_cm")
        # plt.close()
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
    # print(accuracy_score_sets)
    # mean = np.mean(accuracy_score_sets['test'])
    # std_dev = np.std(accuracy_score_sets['test'])
    #
    # print(f"Średnia: {mean:.2f}, Odchylenie standardowe: {std_dev:.2f}")
    return accuracy_score_sets

def main():
    org_train_data = None
    org_test_data = None

    subset_len = 100
    num_epochs = 1000
    if p.dataset2 == 'mnist':
        # org_train_data = importer.import_mnist_subset(True, subset_len)
        # org_test_data = importer.import_mnist(False)
        org_train_data = import_mnist_subset(True, subset_len)
        org_test_data = import_mnist(False)
    elif p.dataset2 == 'imagenette':
        # org_train_data = importer.import_imagenette_subset('train', subset_len)
        # org_test_data = importer.import_imagenette('val')
        org_train_data = import_imagenette_subset('train', subset_len)
        org_test_data = import_imagenette('val')

    train_loader = DataLoader(org_train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(org_test_data, batch_size=32, shuffle=False)

    accuracies = []
    for i in range(2):
        print(f"Iter {i}")
        mlpModel = MNIST_CNNModel(32, 10)
        acc = np.max(train_and_eval_model(mlpModel, train_loader, test_loader, num_epochs)['test'])
        print(f"ACC {subset_len} {num_epochs} {acc}")
        accuracies.append(acc)

    mean = np.mean(accuracies)
    std_dev = np.std(accuracies)
    print(f"Średnia: {mean:.2f}, Odchylenie standardowe: {std_dev:.2f}")

main()

if __name__ == "__main__":
    main()