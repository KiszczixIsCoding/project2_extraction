import torch
import torchvision
from matplotlib import pyplot as plt
from sympy.core.random import shuffle
from torch.utils.data import Subset

from torchvision import datasets
import sklearn
import pandas
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # <- wymusza wspólny rozmiar
    transforms.ToTensor()
])

transform1 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),  # konwertuje do [0, 1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])  # normalizacja jak w ImageNet
])

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

def load_iris():
    return sklearn.datasets.load_iris(return_X_y=False, as_frame=True)

def load_wine():
    return sklearn.datasets.load_wine(return_X_y=False, as_frame=True)

def load_breast_cancer():
    return sklearn.datasets.load_breast_cancer(return_X_y=False, as_frame=True)

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

def import_imagenette(train_str):
    return datasets.Imagenette(root='../resources-imagenette',split=train_str, download=True, transform=transform1)

def unnormalize(tensor, mean, std):
    mean = torch.tensor(mean).view(3, 1, 1)  # Dopasowanie wymiarów
    std = torch.tensor(std).view(3, 1, 1)
    return tensor * std + mean

def import_imagenette_subset(train_str, subset_len):
    imagenette_dataset = datasets.Imagenette(root='../resources-imagenette',split=train_str, download=True, transform=transform_translations_imagenette)
    imagenette_dataset1 = datasets.Imagenette(root='../resources-imagenette',split=train_str, download=True, transform=transform1)

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