import torch
from captum.attr import Lime
from matplotlib import pyplot as plt, gridspec
from skimage.segmentation import slic
import task1.feature_extraction.features_mnist as feature_m

def analyse_lime_mnist(model, org_train_data, org_test_data):
    for index in range(0, 30):
        image, label = org_train_data[index]

        # === 3. Segmentacja z SLIC ===
        image_np = image.squeeze().numpy()  # (28, 28)
        segments = slic(image_np, n_segments=100, compactness=0.1, channel_axis=None, start_label=0)
        feature_mask = torch.tensor(segments.flatten()).unsqueeze(0)
        print(feature_mask.shape)  # (1, 784)
        print(torch.unique(feature_mask))  # np. 0,1,...,9
        # === 4. Interpretacja LIME ===
        # feature_mask = torch.arange(784).reshape(1, 784)

        lime = Lime(model)

        train_data = feature_m.flatten_image(org_train_data)
        test_data = feature_m.flatten_image(org_test_data)

        # train_data = feature_m.centroid_point(org_train_data)
        # test_data = feature_m.centroid_point(org_test_data)

        attributions = lime.attribute(
            inputs=torch.tensor(train_data[0][index]).unsqueeze(0),
            target=int(train_data[1][index]),
            feature_mask=feature_mask,  # maska segmentów (1, H, W)
            n_samples=10000
        )

        # === 5. Wizualizacja wyników ===
        # Przypisujemy atrybucje każdemu segmentowi
        attr_np = attributions.squeeze().detach().numpy().reshape(28, 28)  # (28, 28)
        print(attr_np)

        fig = plt.figure(figsize=(6, 3))
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 0.05], hspace=0.15)

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_title("Oryginał", fontsize=10)
        ax1.imshow(image_np, cmap='gray')
        ax1.axis('off')

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_title("Atrybucje LIME (Captum)", fontsize=10)
        im = ax2.imshow(attr_np, cmap='seismic', interpolation="none")
        ax2.axis('off')

        # Bottom-right: colorbar (cell 2,2)
        cbar_ax = fig.add_subplot(gs[1, 1])
        cbar = plt.colorbar(im, cax=cbar_ax, orientation='horizontal')

        # Bottom-left empty (optional)
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.axis('off')

        plt.tight_layout()
        plt.show()
