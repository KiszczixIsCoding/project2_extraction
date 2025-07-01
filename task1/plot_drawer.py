import datetime
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from matplotlib.path import Path
from scipy.ndimage import gaussian_filter
import matplotlib.colors as mcolors
from scipy.spatial import Voronoi
from sklearn.inspection import DecisionBoundaryDisplay
from torch.utils.data import DataLoader, TensorDataset

from task2.models.Torch2DEstimator import Torch2DEstimator


def draw_linechart(metrics_scores, dataset_name):
    df =  pd.DataFrame(metrics_scores, columns=['adjusted_rand_score', 'homogeneity_score', 'completeness_score'])
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.subplots_adjust(left=0.05, right=0.85)

    title = f"Wykres miar separowalności zbioru {dataset_name} i cechy 'odległość punktów skrajnych'"
    df[['adjusted_rand_score', 'homogeneity_score', 'completeness_score']].plot(kind='line',
                                                                                            grid='true',
                                                                                            figsize=(14, 8),
                                                                                            ax=ax)
    plt.xlabel('epochs')
    plt.ylim(0, 0.8)
    # for index in range(0, epochs):
    #     # eps = 0.1 * index
    #     if index % mod_div == 0:
    #         eps = factor * index
    #         # plt.text(eps, 0.1, f"{n_clusters_list[index - 1]}", fontsize=9)
    #         print(metrics_scores['silhouette_score'])
    #         # plt.text(eps, metrics_scores.sort_values(by='silhouette_score')['silhouette_score'].tolist()[1], f"{n_clusters_list[index]}", fontsize=9)
    #         print(index - start)
    #         plt.text(eps, n_clusters_y, f"{n_clusters_list[index - start]}", fontsize=9)
    ax.set_title(title, pad=25)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()


def draw_accuracy(accuracy_score_sets, dataset_name, iter_folder_path, epochs):
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.subplots_adjust(left=0.05, right=0.85)
    plt.ylim(0.2, 1.01)

    x_labels_list = list(range(0, epochs))
    plt.plot(x_labels_list, accuracy_score_sets['train'], label='Train accuracy', color='blue')
    plt.plot(x_labels_list, accuracy_score_sets['test'], label='Test accuracy', color='orange')

    formatted_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # results_folder_path = f'{iter_folder_path}/{parameter}/accuracy_linechart/'
    #
    # if not os.path.exists(results_folder_path):
    #     os.makedirs(results_folder_path)

    # save_path = os.path.join(results_folder_path, f'{formatted_timestamp}.png')
    # ax.set_title(f"Wykres miary accuracy na zbiorach treningowym i testowym dla danych ze zbioru {dataset_name} oraz sposobem ekstrakcji 'wszystkie piksele'", pad=25)
    ax.set_title(
        f"Wykres miary accuracy na zbiorach treningowym i testowym dla danych ze zbioru {dataset_name} oraz sposobem ekstrakcji 'symetria + wariancja",
        pad=25)
    ax.set_xlabel("epochs")
    plt.grid(True, linestyle='--', color='gray')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(iter_folder_path)
    # plt.show()
    plt.close()


def voronoi_finite_polygons_2d(vor, radius=None):
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = np.ptp(vor.points).max()

    # Construct a map containing all ridges for a given point
    all_ridges = defaultdict(list)
    # all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


def is_point_in_polygon(points, polygon):
    path = Path(polygon)
    return path.contains_point(points)


def draw1(data, pred_labels, scope):
    # np.random.seed(1234)
    # points = np.random.rand(15, 2)
    # colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'magenta', 'pink', 'gray']
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'magenta', 'pink', 'gray', 'brown'] + list(
        mcolors.XKCD_COLORS.values())
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.subplots_adjust(left=0.05, right=0.85)

    # compute Voronoi tesselation
    print(data[['feature_1', 'feature_2']])
    vor = Voronoi(data[['feature_1', 'feature_2']])
    regions, vertices = voronoi_finite_polygons_2d(vor)
    # plt.xlabel('Symetria vert./hor.')
    # plt.ylabel('Wariancja w osi Y')

    ind = 0
    for point_idx, region in enumerate(regions):
        polygon = vertices[region]
        ind += 1
        ax.fill(*zip(*polygon), alpha=0.4, facecolor=colors[int(pred_labels[point_idx])],
                edgecolor=colors[int(pred_labels[point_idx])], linestyle='solid')

    # for point_idx, region in enumerate(regions):
    #     polygon = vertices[region]
    #     point_index = None
    #     for index, row in data.iterrows():
    #         if is_point_in_polygon((row['feature_1'], row['feature_2']), polygon):
    #             point_index = index
    #             print(point_idx)
    #     ax.fill(*zip(*polygon), alpha=0.4, facecolor=colors[int(pred_labels[point_index])], edgecolor=colors[int(pred_labels[point_index])], linestyle='solid')
    ax.set_title(
        f"Diagram woronoja dla danych ze zbioru mnist oraz sposobem ekstrakcji 'symetria + wariancja",
        pad=25)
    for cluster in range(0, len(list(set(data['label'])))):
        filtered_points = data[data['label'] == cluster]
        ax.plot(filtered_points['feature_1'], filtered_points['feature_2'], 'o', color=colors[cluster], markersize=5,
                alpha=0.4)

    # plt.plot(points[:,0], points[:,1], 'ko')
    plt.xlim(vor.min_bound[0] - 0.02, vor.max_bound[0] + 0.02)
    plt.ylim(vor.min_bound[1] - 0.02, vor.max_bound[1] + 0.02)

    # formatted_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # results_folder_path = f'./results/{p.method}/artificial_sets/voronoi/{dataset_name}/{len(list(set(pred_labels)))}/'
    #
    # if not os.path.exists(results_folder_path):
    #     os.makedirs(results_folder_path)

    # save_path = os.path.join(results_folder_path, f'{formatted_timestamp}.png')
    labels = [str(i) for i in range(len(list(set(pred_labels))))]
    legend_handles = [
        Patch(facecolor=colors[i % len(colors)], label=labels[i])
        for i in range(len(list(set(pred_labels))))
    ]

    # Add the legend to your existing plot
    ax.legend(handles=legend_handles, title="Klaster", loc='center left', bbox_to_anchor=(1, 0.5))
    # ax.set_title(f"Diagram Woronoja dla sztucznie wygenerowanego zbioru {dataset_name} i metody {p.method} przy parametrze n_clusters równym {len(list(set(pred_labels)))}", pad=25)
    plt.savefig(scope)
    plt.close()
    # plt.show()

    return "CDE"

def draw3(data, pred_labels, clf, scope):
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'magenta', 'pink', 'gray', 'brown'] + list(
        mcolors.XKCD_COLORS.values())
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.subplots_adjust(left=0.05, right=0.85)

    # for label in np.unique(data['label']):
    #     subset = data[data['label'] == label]
    #     print(f"Klasa {label}:")
    #     print(f" feature_1 min/max: {subset['feature_1'].min()}/{subset['feature_1'].max()}")
    #     print(f" feature_2 min/max: {subset['feature_2'].min()}/{subset['feature_2'].max()}")

    # # Create a mesh to plot
    # h = .02  # step size in the mesh
    # datax_min, datax_max = data['feature_1'].min() - 0.02, data['feature_1'].max() + 0.02
    # datay_min, datay_max = data['feature_2'].min() - 0.02, data['feature_2'].max() + 0.02
    # data_xx, data_yy = np.meshgrid(np.arange(datax_min, datax_max, 0.01), np.arange(datay_min, datay_max, 0.01))

    data_xx, data_yy = np.meshgrid(
    np.linspace(data['feature_1'].min(), data['feature_1'].max(), 2000),
        np.linspace(data['feature_2'].min(), data['feature_2'].max(), 2000)
    )

    grid = np.vstack([data_xx.ravel(), data_yy.ravel()]).T
    grid_tensor = torch.tensor(grid, dtype=torch.float32)
    grid_loader = DataLoader(TensorDataset(grid_tensor), batch_size=32, shuffle=False)

    loader_preds = []
    test_batch = 0
    # with torch.no_grad():
    #     for batch in grid_loader:
    #         test_batch += 32
    #         print(test_batch)
    #         x = batch[0]
    #         y_batch = clf.predict(x)
    #         # print(y_batch)
    #         loader_preds.append(y_batch)
    # #
    # y_preds_merged = torch.cat(loader_preds).numpy()
    # y_pred_grid = y_preds_merged.reshape(data_xx.shape)

    # y_pred = np.reshape(clf.predict(torch.tensor(grid, dtype=torch.float32)), data_xx.shape)
    # my_y_preds = clf.predict(torch.tensor(data[['feature_1', 'feature_2']].to_numpy(), dtype=torch.float32))

    # print("DATA:")
    # print(data)
    # print("GRID:")
    # print(grid)
    # print(len(my_y_preds))
    # print(y_pred.shape)
    # print(f"Unique {np.unique(y_pred_grid)}")
    # print(f"Unique {np.unique(my_y_preds)}")
    # print(f"Unique {np.unique(y_pred)}")

    # Z = clf.predict(torch.tensor(np.c_[data_xx.ravel(), data_yy.ravel()], dtype=torch.float32))

    # display = DecisionBoundaryDisplay(xx0=data_xx, xx1=data_yy, response=y_pred)
    te = Torch2DEstimator(clf)
    te.fitted_ = True

    display = DecisionBoundaryDisplay.from_estimator(te, data[['feature_1', 'feature_2']].to_numpy(), response_method="predict")

    print(type(display))
    # display = DecisionBoundaryDisplay(xx0=np.vstack((data_xx, data['feature_1'].to_numpy())), xx1=np.vstack((data_yy, data['feature_2'].to_numpy())), response=np.vstack((y_pred_grid, pred_labels)))

    display.plot(cmap=mcolors.ListedColormap(['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'magenta', 'pink', 'gray', 'brown']), alpha=0.4, ax=ax)

    # plt.contour(
    # data_xx, data_yy, y_pred,
    #     colors='black',  # kolor linii
    #     linewidths=3  # grubość linii
    # )
    # print(data_xx.shape)
    # print(data_yy.shape)
    # Predict class labels for each point in the mesh


    # Put the result into a color plot
    # plt.contourf(data_xx, data_yy, Z, cmap=mcolors.ListedColormap(
    #     ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'magenta', 'pink', 'gray', 'brown']), alpha=0.4)
    #
    # contour = plt.contour(data_xx, data_yy, Z, levels=[0.5], colors='black', linewidths=3)

    for cluster in range(0, len(list(set(data['label'])))):
        filtered_points = data[data['label'] == cluster]
        plt.plot(filtered_points['feature_1'], filtered_points['feature_2'], 'o', color=colors[cluster], markersize=2)

    labels = [str(i) for i in range(len(list(set(pred_labels))))]
    legend_handles = [
        Patch(facecolor=colors[i % len(colors)], label=labels[i])
        for i in range(len(list(set(pred_labels))))
    ]
    ax.legend(handles=legend_handles, title="Klaster", loc='center left', bbox_to_anchor=(1, 0.5))

    # plt.xlim(datax_min, datax_max)
    # plt.ylim(datay_min, datay_max)

    plt.title(f"Diagram Woronoja dla sztucznie wygenerowanego zbioru  i metody  dla liczby sasiadów równej równym", pad=25)
    # formatted_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    #
    # results_folder_path = f'{iter_folder_path}/{parameter}/voronoi/'
    #
    # if not os.path.exists(results_folder_path):
    #     os.makedirs(results_folder_path)
    #
    # save_path = os.path.join(results_folder_path, f'{accuracy_type}_{formatted_timestamp}.png')
    plt.savefig(scope)
    plt.close()
