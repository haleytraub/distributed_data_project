import unittest
import numpy as np

from ParallelPCA import generate_dataset, iris_data, manual_PCA, perform_PCA, divide_data, combine_PCA, parallel_PCA

def test_iris():
    data = iris_data()
    data.shape == (150, 4) ##making sure it is reading in the correct data set and looking at the shape

def test_divide_data():
    data = generate_dataset(100, 5)
    subsets = divide_data(data, 4)
    assert len(subsets) == 4, "Data should be divided into 4 subsets"
    assert sum(subset.shape[0] for subset in subsets) == data.shape[0], "Total number of samples should be the same after division"
    assert all(subset.shape[1] == data.shape[1] for subset in subsets), "Each subset should have the same number of features"

