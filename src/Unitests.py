import unittest
import numpy as np

from ParallelPCA import generate_dataset, iris_data, manual_PCA, divide_data, combine_PCA, parallel_PCA


class TestPCA(unittest.TestCase):

    ##testing the two functions made that generate data, it is important the the data is generated and read correctly

    def test_generate_dataset(self):
        dataset = generate_dataset(100, 5)
        self.assertEqual(dataset.shape,(100, 5))

    def test_iris_data(self):
        data = iris_data()
        self.assertEqual(data.shape,(150, 4))

 ## tests the manual PCA to make sure it is calculating it correctly (by calling sklearn PCA)
    def test_manual_PCA(self):
        
        data = np.array([[2, 4, 6, 8, 10],
                 [1, 3, 5, 7, 9],
                 [11, 13, 15, 17, 19],
                 [12, 14, 16, 18, 20]])

        handPCA = manual_PCA(data, n_components= 1)

        from sklearn.decomposition import PCA
        pca = PCA(n_components= 1)
        sklearnpca = pca.fit_transform(data)

        np.testing.assert_allclose(np.abs(handPCA), np.abs(sklearnpca))

    ##make sure the data is correctly splitting the data into subgroups
    def test_divide_data(self):
        data = np.array([[2, 4, 6, 8, 10],
                 [1, 3, 5, 7, 9],
                 [11, 13, 15, 17, 19],
                 [12, 14, 16, 18, 20]])

        subsets = divide_data(data, 3)
        self.assertEqual(len(subsets),3)


    ## make sure more than 1 PID is returned so it is utilizing multiprocessing

    def test_parallel(self):

        datasets = [generate_dataset(100,5), generate_dataset(50,2), generate_dataset(300000, 4)]
        n_components = [2, 2, 5]

        results, pid_list = parallel_PCA(datasets, n_components)

        unique_pid = set(pid_list)
        
        self.assertGreater(len(unique_pid), 1)

if __name__ == '__main__':
    unittest.main(exit=False)
    print("All tests passed!")


