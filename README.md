# distributed_data_project

## 1. Algorithm Purpose
Parallel PCA is used for a multitude of reasons: 
* **Handling large datasets**: PCA can become inefficient and even infeasible for large datasets. This is due to the fact PCA can struggle with memory and computational constraints when dealing with large datasets. Parallel PCA is a solution to this problem as it distributes the computational efforts across multiple processors.
* **Computational Efficiency/ Scalability**: Adding parallelization to PCA can reduce the time it takes for computation. Parallel PCA is also designed to scale with the amount of data and the number of available computational resources. Both of these aspects can be extremely helpful when dealing with high dimensional or large datasets. 
* **Application**: Parallel PCA can be utilized in many different fields to include image processing, finance, etc. Any domain where there is a need for dimensionality reduction, feature extraction or dara visualization, Parallel PCA can be utilized.
## 2. Hyperparameters

## 3. Background
Principal component analysis is an unsupervised learning dimensionality reduction technique. The goal is to reduce the data from their original high- dimensional feature space to a new set of components or principal component axes. The algorithm utilizes parallel processing by first splitting the dataset into local subsets. After this is done, each process will perform two tasks to include PCA calculation and covariance matrix computation. The reasoning behind this strategy is to mitigate the amount of time it takes to sort through large datasets when performing operations. By distributing the dataset across multiple processes, it could reduce the processing time. Once the PCA and covariance matrix are completed on each subset, the results are combined to produce a final PCA output. The final PCA output represents the reduced dimension of the original dataset, which will be achieved by the utilization of distributed and parallel processing techniques. 

### 3a. History


### 3b. Variations

## 4. Pseudo code

1. Initialize MPI
2. Get rank and size of the MPI Communicator
3. If rank == 0
   
      Load the dataset
   
      Standardize the dataset
   
      Split the data into subsets
   
5. Open the channel from the dataset to the processes
6. Split the dataset into local datasets for each process
7. For each process:

    Perform PCA on the local dataset

    Compute the covariance matrix of the dataset

9. Sum the covariance matrix from each process
10. Sum the size of all the local datasets 
11. Divide the summed covariance matrix by the total number of samples
12. Compute the eigenvectors and eigenvalues of the total covariance matrix
13. Sort the eigenvectors based on the eigenvalues in descending order
14. Select the top k eigenvectors to retain
15. Broadcast the selected eigenvectors to all processes
16. Transform data (projecting the local subsets on selected eigenvectors)
17. Reconstruct the global transformed dataset (gathered from transformed data)
18. Finalize MPI


## 5. Example code to import and use module

## 6. Visualization or animation of algorithm steps or results

## 7. Benchmark Results

Comparison of efficiency and effectiveness 

## 8. Lessons Learned
Such as new code snippets to support some computations

## 9. Unit-testing strategy
What steps of the algorithm were tested individually?
Code-coverage measurement
