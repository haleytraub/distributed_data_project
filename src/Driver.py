
from ParallelPCA import generate_dataset, iris_data, parallel_PCA
import pandas as pd

if __name__ == "__main__":
    dataset1 = generate_dataset(10000, 5)
    dataset2 = generate_dataset(300000, 6)
    iris = iris_data()

    datasets = [dataset1, dataset2, iris]
    n_components_list = [2, 2, 2]

    results = parallel_PCA(datasets, n_components_list)

    pca_df_1 = pd.DataFrame(results[0], columns=['PC1', 'PC2'])
    pca_df_2 = pd.DataFrame(results[1], columns=['PC1', 'PC2'])
    pca_df_3 = pd.DataFrame(results[2], columns=['PC1', 'PC2'])

    print("PCA Result for Random Dataset 1:", pca_df_1.head())
    print("\nPCA Result for Random Dataset 2:", pca_df_2.head())
    print("\nPCA Result for Iris Dataset:", pca_df_3.head())
