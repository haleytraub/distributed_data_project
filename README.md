# distributed_data_project

## 1. Algorithm Purpose

## 2. Hyperparameters

## 3. Background

### 3a. History

### 3b. Variations

## 4. Pseudo code

1. Initialize MPI
2. Get rank and size of the MPI Communicator
3. If rank == 0 \n
      Load the dataset \n
      Standardize the dataset \n
      Split the data into subsets \n
4. Open the channel from the dataset to the processes
5. Split the dataset into local datasets for each process
6. For each process: 
    Perform PCA on the local dataset 
    Compute the covariance matrix of the dataset 
7. Sum the covariance matrix from each process
8. Sum the size of all the local datasets 
9. Divide the summed covariance matrix by the total number of samples
10. Compute the eigenvectors and eigenvalues of the total covariance matrix
11. Sort the eigenvectors based on the eigenvalues in descending order
12. Select the top k eigenvectors to retain
13. Broadcast the selected eigenvectors to all processes
14. Transform data (projecting the local subsets on selected eigenvectors)
15. Reconstruct the global transformed dataset (gathered from transformed data)
16. Finalize MPI


## 5. Example code to import and use module

## 6. Visualization or animation of algorithm steps or results

## 7. Benchmark Results

Comparison of efficiency and effectiveness 

## 8. Lessons Learned
Such as new code snippets to support some computations

## 9. Unit-testing strategy
What steps of the algorithm were tested individually?
Code-coverage measurement
