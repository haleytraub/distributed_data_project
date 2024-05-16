from mpi4py import MPI
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#random dataset generated
np.random.seed(0)
num_samples = 100
num_features = 5
dataset = np.random.rand(num_samples, num_features)

# Standardize the dataset
scaler = StandardScaler()
scaled_dataset = scaler.fit_transform(dataset)

if rank == 0:
    subset_size = len(scaled_dataset) // size
    subsets = [scaled_dataset[i * subset_size: (i + 1) * subset_size] for i in range(size)]
else:
    subsets = None

local_subset = comm.scatter(subsets, root=0)

pca = PCA()
local_pca_result = pca.fit_transform(local_subset)

# Compute the covariance matrix of the local dataset
local_covariance = np.cov(local_pca_result, rowvar=False)

# Sum the covariance matrix from each process
global_covariance = comm.reduce(local_covariance, op=MPI.SUM, root=0)

# Sum the size of all the local datasets
total_samples = comm.reduce(len(local_subset), op=MPI.SUM, root=0)

if rank == 0:
    # Divide the summed covariance matrix by the total number of samples
    global_covariance /= total_samples

    # Compute the eigenvectors and eigenvalues of the total covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(global_covariance)

    # Sort the eigenvectors based on the eigenvalues in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # Select the top k eigenvectors to retain
    k = 2
    selected_eigenvectors = sorted_eigenvectors[:, :k]

else:
    selected_eigenvectors = None

# Broadcast the selected eigenvectors to all processes
selected_eigenvectors = comm.bcast(selected_eigenvectors, root=0)

# Transform data (projecting the local subsets on selected eigenvectors)
local_transformed_data = np.dot(local_subset, selected_eigenvectors)

# Reconstruct the global transformed dataset (gathered from transformed data)
transformed_data = comm.gather(local_transformed_data, root=0)

if rank == 0:
    reconstructed_data = np.concatenate(transformed_data, axis=0)
    # Finalize MPI
    MPI.Finalize()
    print(reconstructed_data.shape)
