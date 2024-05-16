# distributed_data_project

## 1. Algorithm Purpose

## 2. Hyperparameters

## 3. Background

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
