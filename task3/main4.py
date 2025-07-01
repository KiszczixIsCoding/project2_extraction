import torch
import torch.nn as nn
import torch.optim as optim
from captum.metrics import sensitivity_max
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from captum.attr import Saliency
import numpy as np
import matplotlib.pyplot as plt
import captum.attr import Lime

# Load and preprocess the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names

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

# Define a simple neural network
class IrisNet(nn.Module):
    def __init__(self):
        super(IrisNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 10),
            nn.ReLU(),
            nn.Linear(10, 3)
        )

    def forward(self, x):
        return self.net(x)

model = IrisNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train the model
for epoch in range(100):
    optimizer.zero_grad()
    output = model(X_train_tensor)
    loss = criterion(output, y_train_tensor)
    loss.backward()
    optimizer.step()

print("Training complete. Test accuracy:",
      (model(X_test_tensor).argmax(1) == y_test_tensor).float().mean().item())

# Choose a test input to explain
test_input = X_test_tensor[0].unsqueeze(0)  # add batch dimension
test_input.requires_grad_()
pred_class = model(test_input).argmax().item()

# # Use Captum to compute saliency
# saliency = Saliency(model)
# all_attributions = []
# for i in range(len(X_test_tensor)):
#     x = X_test_tensor[i].unsqueeze(0)
#     x.requires_grad_()
#     pred = model(x).argmax().item()
#     attr = saliency.attribute(x, target=pred)
#     all_attributions.append(attr.squeeze().detach().numpy())
#
# # Uśrednij saliency po przykładach
# avg_saliency = np.mean(np.array(all_attributions), axis=0)
#
# # Wykres
# plt.bar(feature_names, avg_saliency)
# plt.title("Average Saliency Across Test Set")
# plt.ylabel("Average attribution")
# plt.show()
infidelity(net, perturb_fn, input_tensor, attribution)
sensitivity_max(saliency.attribute, input_tensor, target = 3)

