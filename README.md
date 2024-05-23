##Parallel PCA

## 1. Algorithm Purpose
   Parallel PCA is used for a multitude of reasons: 
   * **Handling large datasets**: PCA can become inefficient and even infeasible for large datasets. This is       due to the fact PCA can struggle with memory and computational constraints when dealing with large datasets.    Parallel PCA is a solution to this problem as it distributes the computational efforts across multiple          processors.
   * **Computational Efficiency/ Scalability**: Adding parallelization to PCA can reduce the time it takes for    computation. Parallel PCA is also designed to scale with the amount of data and the number of available         computational resources. Both of these aspects can be extremely helpful when dealing with high dimensional      or large datasets. 
   * **Application**: Parallel PCA can be utilized in many different fields to include image processing,          finance, etc. Any domain where there is a need for dimensionality reduction, feature extraction or data       
   visualization, Parallel PCA can be utilized.
## 2. Hyperparameters

## 3. Background
      Principal component analysis is an unsupervised learning dimensionality reduction technique. The goal is    to reduce the data from their original high- dimensional feature space to a new set of components or          principal component axes. The algorithm utilizes parallel processing by first splitting the dataset into local subsets. After this is done, each process will perform two tasks to include PCA calculation and covariance matrix computation. The reasoning behind this strategy is to mitigate the amount of time it takes to sort through large datasets when performing operations. By distributing the dataset across multiple processes, it could reduce the processing time. Once the PCA and covariance matrix are completed on each subset, the results are combined to produce a final PCA output. The final PCA output represents the reduced dimension of the original dataset, which will be achieved by the utilization of distributed and parallel processing techniques. 

### 3a. History
In the 1900s, PCA was developed as a statistical technique for data analysis and dimensionality reduction. In the 1980- 1990s, parallel computing was introduced and the concepts began to merge. MPI standardization became more prevalent which set the stage for Parallel PCA. Moving into the 2000s, there became an emphasis on big data frameworks and Apache Spark. With all of these ideas, the current trends are focused on scalability and efficiency. 

### 3b. Variations
Variations of PCA can be seen below:
* Covariance - Matrix Approach: This is the variation I will be doing. This variation includes computing the covariance matrix of the dataset in parallel by distributing the data among multiple processors. 
* Single Vector Decomposition: Uses SVD to decompose the data matrix directly. In regards to the parallelism, it will distribute the data among the processors and each will compute the decomposition.
* MapReduce based PCA: Utilizes Hadoop MapReduce to split data into chunks processed in parallel. The partial results are aggregated in the Reduce phase.
* Stochastic PCA: Approximates principal components using randomization, processing data in smaller chunks in parallel.

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

Please refer to the other coding modules in this repository for an example import/ use module.

## 6. Visualization or animation of algorithm steps or results



## 7. Benchmark Results

Comparison of efficiency and effectiveness 

## 8. Lessons Learned
Such as new code snippets to support some computations

## 9. Unit-testing strategy
What steps of the algorithm were tested individually?
Code-coverage measurement
