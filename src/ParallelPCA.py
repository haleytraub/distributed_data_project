import numpy as np 
import pandas as pd
from multiprocessing import Pool, cpu_count
from sklearn.datasets import load_iris
import os
import time

def manual_PCA(data, n_components = 2):
    mean = np.mean(data, axis=0)
    center_mean = data - mean 

    covariance_matrix = np.cov(center_mean, rowvar = False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    sort_index = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors= eigenvectors[:, sort_index]

    eigenvector_subset = sorted_eigenvectors[:, 0:n_components]
    transformed_data = np.dot(eigenvector_subset.transpose(), center_mean.transpose()).transpose()

    return transformed_data

def perform_PCA(data, n_components):
    print(f"Process ID: {os.getpid()} is processing data subset")
    transformed_data = manual_PCA(data, n_components)
    return transformed_data

def divide_data(data, subset):
    return np.array_split(data, subset)

def combine_PCA(subsets):
    return np.concatenate(subsets, axis = 0)

def generate_dataset(n_samples, n_features):
    return np.random.rand(n_samples, n_features)

def iris_data():
    iris = load_iris()
    return iris.data


def parallel_PCA(datasets, n_component_list):
    subset = cpu_count()
    results = []

    with Pool(cpu_count()) as pool:
        start_time = time.time()
        for i, data in enumerate(datasets):
            subsets = divide_data(data, subset)
            pca_result = pool.starmap(perform_PCA, [(subset, n_component_list[i]) for subset in subsets])
            combine_results = combine_PCA(pca_result)
            results.append(combine_results)
        parallel_time = time.time() - start_time

    print(f"Parallel processing took {parallel_time:.2f} seconds")
    return results