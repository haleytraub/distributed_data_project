
from ParallelPCA import generate_dataset, iris_data, parallel_PCA, plot_results, manual_PCA
import pandas as pd
import time

if __name__ == "__main__":
    dataset1 = generate_dataset(10000, 10)
    dataset2 = generate_dataset(300000, 6)
    iris = iris_data()

    datasets = [dataset1, dataset2, iris]
    dataset_titles = ["Random Dataset 1", "Random Dataset 2", "Iris Dataset"]
    n_components_list = [2, 2, 2]

    results, pid_list = parallel_PCA(datasets, n_components_list)

    start_time = time.time()
    regular_times = []
    for data in datasets:
        manual_PCA(data, n_components=2)
    total_regular_time = time.time() - start_time

    print(f"Total time for regular PCA on all datasets: {total_regular_time:.6f} seconds")

    plot_results(results[0], "PCA Result for Random Dataset 1")
    plot_results(results[1], "PCA Result for Random Dataset 2")
    plot_results(results[2], "PCA Result for Iris Dataset")
