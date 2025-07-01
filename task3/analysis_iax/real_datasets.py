import random

import numpy as np
import torch
from captum.attr import Saliency, IntegratedGradients, NoiseTunnel, GuidedBackprop, DeepLift
from captum.metrics import infidelity, sensitivity_max
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn, optim
from sklearn.datasets import load_iris, load_wine, load_breast_cancer

from task1.models.model import MLPModel


def perturb_fn(inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Dodaj delikatny szum do każdej cechy (np. 2% wartości)
    noise = torch.randn_like(inputs) * 0.005  # np. 2% standard deviation
    perturbed = inputs + noise
    return perturbed, noise

def analyse_real_datasets():
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names

    model = MLPModel(X.shape[1], 32, len(set(y)))
    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    for epoch in range(60):
        print(epoch)
        optimizer.zero_grad()
        output = model(X_train_tensor)
        loss = criterion(output, y_train_tensor)
        loss.backward()
        optimizer.step()

        print("Training complete. Test accuracy:", (model(X_test_tensor).argmax(1) == y_test_tensor).float().mean().item())

    # Choose a test input to explain
    test_input = X_test_tensor[0].unsqueeze(0)  # add batch dimension
    test_input.requires_grad_()
    pred_class = model(test_input).argmax().item()

    # Use Captum to compute saliency
    #
    # ig = IntegratedGradients(model)
    # nt = NoiseTunnel(saliency)



    attr = None

    x = None
    infid_vals = []
    sensitivity_vals = []
    stats = {
        "saliency": {},
        "ig": {},
        "nt-saliency": {},
        "nt-ig": {},
        "gbp": {},
        "dp": {}
    }

    for method_name in ["saliency", "ig", "gbp", "nt-saliency", "nt-ig", "dp"]:
        all_attributions = []
        for i in range(len(X_test_tensor)):
            x = X_test_tensor[i].unsqueeze(0)
            x.requires_grad_()
            pred = model(x).argmax().item()

            method = None
            if method_name == "saliency":
                method = Saliency(model)
                attr = method.attribute(x, target=pred)
            elif method_name == "ig":
                method = IntegratedGradients(model)
                attr = method.attribute(x, target=pred, n_steps=50, baselines=torch.zeros_like(x))
            elif method_name == "gbp":
                method = GuidedBackprop(model)
                attr = method.attribute(x, target=pred)
            elif method_name == "nt-saliency":
                method = NoiseTunnel(Saliency(model))
                attr = method.attribute(x, nt_type='smoothgrad', stdevs=0.05, nt_samples=100, target=pred)
            elif method_name == "nt-ig":
                method = NoiseTunnel(IntegratedGradients(model))
                attr = method.attribute(x, nt_type='smoothgrad', stdevs=0.05, nt_samples=100, target=pred)
            elif method_name == "dp":
                method = DeepLift(model)
                attr = method.attribute(x, target=pred)
            global_attr = attr
            infid = infidelity(model, perturb_func=perturb_fn, inputs=x, attributions=global_attr, target=pred)
            sensitivity = sensitivity_max(
                explanation_func=method.attribute,
                inputs=x,
                target=pred,
                n_perturb_samples=20,
                perturb_radius=0.01
            )

            infid_vals.append(infid.item())
            sensitivity_vals.append(sensitivity.item())
            all_attributions.append(attr.detach().numpy().flatten())

        stats[method_name] = {
            "infidelity": np.mean(infid_vals),
            "sensitivity": np.mean(sensitivity_vals)
        }

        # Wykres
        avg_attributions = np.mean(np.array(all_attributions), axis=0)
        sorted_indices = np.argsort(avg_attributions)

        # Wybierz 3 najniższe i 3 najwyższe (bez powtarzania)
        selected_indices = list(sorted_indices[:3]) + list(sorted_indices[-3:])

        # Przygotuj dane do wykresu
        selected_features = [feature_names[i].replace(" ", "\n") for i in selected_indices]
        selected_attributions = [avg_attributions[i] for i in selected_indices]

        colors = ['tomato'] * 3 + ['mediumseagreen'] * 3
        # if method_name == "gbp":
        print(method_name)
        # plt.bar(selected_features, selected_attributions, color=colors)
        # plt.ylim(-0.125, 0.125)
        # plt.gca().yaxis.set_major_locator(MultipleLocator(0.025))
        # plt.axhline(0, color='black', linewidth=0.8)
        # plt.grid(axis='y', linestyle='--', alpha=0.6)
        # plt.title("Średnie dla najgorszych i najlepszych atrybucji z Saliency na zbiorze iris", fontsize=10, pad=15)
        # plt.ylabel("Średnia atrybucja", fontsize=10)
        # plt.xticks(rotation=30, ha='right', fontsize=8)
        # plt.subplots_adjust(bottom=0.25)
        # plt.show()

        plt.bar(feature_names, avg_attributions)
        plt.ylim(0, 0.4)
        # plt.gca().yaxis.set_major_locator(MultipleLocator(0.025))
        plt.axhline(0, color='black', linewidth=0.8)
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.title("Średnie dla atrybucji z Saliency na zbiorze iris", fontsize=10, pad=15)
        plt.ylabel("Średnia atrybucja", fontsize=10)
        plt.xticks(rotation=30, ha='right', fontsize=8)
        plt.subplots_adjust(bottom=0.25)
        plt.show()

    print(stats)
    # Uśrednij saliency po przykładach
    # avg_saliency = np.mean(np.array(all_attributions), axis=0)
    #
    # # Wykres

    # plt.bar(feature_names, avg_saliency)
    # plt.title("Average Saliency Across Test Set")
    # plt.ylabel("Average attribution")
    # plt.show()

    # sensitivity_max(saliency.attribute, input_tensor, target=3)
analyse_real_datasets()