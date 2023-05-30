# Iris-and-Image-Clustering-using-KMeans-Clustering

**PART 1 - IRIS CLUSTERING**

This is the famous Iris dataset and serves as an easy benchmark for evaluation. Test your K-Means Algorithm on this easy dataset with 4 features:
- sepal length in cm
- sepal width in cm
- petal length in cm
- petal width in cm
- and 150 instances.

Essentially assign the 150 instances in the test file to 3 cluster IDs given by 1, 2 or 3. The leaderboard will output the V-measure and this benchmark can be used as an easy step. The training data is a NULL FILE. The file test-data-iris.txt, under "Test data," contains the data you use for clustering. The format example is given by format-file-iris.txt.

**PART 2 - IMAGE CLUSTERING**

**Overview and Assignment Goals:**

- The objectives of this assignment are the following:
- Implement the K-Means Algorithm
- Deal with Image data (processed and stored in vector format)
- Think about Best Metrics for Evaluating Clustering Solutions

**Detailed Description:**

Here, we are required to implement the K-Means algorithm on your own. You are NOT allowed to use libraries for this assignment except for pre-processing. Input Data (provided under Test) consists of 10,000 images of handwritten digits (0-9). The images were scanned and scaled into 28x28 pixels. For every digit, each pixel can be represented as an integer in the range [0, 255] where 0 corresponds to the pixel being completely white, and 255 corresponds to the pixel being completely black. This gives us a 28x28 matrix of integers for each digit. We can then flatten each matrix into a 1x784 vector. No labels are provided. Format of the input data: Each row is a record (image), which contains 784 comma-delimited integers.

**Problem statement and Analysis of K Means Algorithm:**

K Means Clustering: K means is an unsupervised learning algorithm with no target labels. By using this we can calculate the V1 score, and it is used for clustering data using k clusters without the labels given.

- **Iris Clustering** – In the assignment, 4 different features like sepal length and width and petal length and width has been given and with no target label. The aim is to cluster those such that the distance between the points and centroid is less and such that the outer distance between 2 centroids is more. We can say that k value as 3 is an optimal value when finding the V1 score using the Elbow method and as given in the problem statement. V1 score accuracy has been calculated as 0.80 in miner after performing the experiment.

- **Image clustering** – In this assignment, the Input Data has 10,000 images of handwritten digits (0 - 9) and scaled into 28 x 28 pixels. For every digit, each pixel can be represented as an integer in the range [0, 255]. This gives us a 28 x 28 matrix of integers for each digit. We can then flatten each matrix into a 1 x 784 vector with no labels provided. We can say that the optimal value of K as asked to consider in the homework as 10 and the V1 score obtained is 0.69 after performing the experiment.

**Analysis of approach for K Means:** K means finds the k clusters or groups of data clustered by the data given in the test dataset. Here, we find the centroids between the data points, and we find the closest point to each centroid using the distance metric which is Euclidean distance, cosine similarity and Manhattan distance (using libraries listed below and approach). The new centroid will be found which is the mean value for each group or cluster. We repeat till we find the same or ideal centroid. We calculate the distance between old centroids and new centroids using norm Distance function and if the distances are smaller then we knew that we reached the final point of centroids and stop the algorithm. Then a graph can be plotted to view the clusters and evaluate the V1 score changes using different k values.

**APPROACH FOLLOWED:**

**Part 1 – Iris Clustering**

1. The dataset given for iris clustering will be loaded into a data frame using pandas with 4 features as 4 column names like Sepal Length, Sepal Width, Petal Length and Petal Width.
2. Then they are preprocessed with minmax scalar and any rows with only 0s are removed as a part of preprocessing and a null check is done to check if there are any null values present in the dataset. Minmax scalar has been used to preprocess the data as it’s the best approach where all data in the data frame gets normalized from 0 to 1.
3. A plot has been made for visualizing the dataset of Sepal Length and Sepal Width for better understanding to perform clustering. And also a table with the data imported has also been shown below as a view with 4 different feature for the dataset.
4. Feature selection like PCA or SVM can be done to reduce the dimensionality but in this experiment performed it reduced the V1 score and as there are only less data, we don’t need to use it.
5. Then we find the centroids and Euclidean distance between the datapoints to perform the experiment.
6. Here, we initialize the centroid with 7 as the runs or iterations. Now we choose random 7
set of centroids to calculate the inner distances among them. We chose the iterations as 7 for initializing centroids as it gave us better results (different values for iter like 1-10 has been considered in the experiment). Then we calculate the distance between the old centroids and new centroid.
7. The distance used here is Euclidean distance. But we can used different approaches to calculate the distance between 2 datapoints like Euclidean distance, cosine similarity and Manhattan distance using the libraries given below in the report. Best results (score) are shown for Euclidean distance comparatively, so it has been choosed.

<img width="522" alt="image" src="https://github.com/sanjana-govindu/Iris-and-Image-Clustering-using-KMeans-Clustering/assets/54507596/976cbac0-b83c-4cb6-841e-04ced48ef748">

8. Now, with every datapoint in the dataset the distance is calculated with the centroid and the one which is nearer to the centroid is where the data or element is considered to be clustered into.
9. The new centroid value is the mean or average value of all the datapoints in the cluster formed.
10. Then the above two steps are performed repetitively until the centroid doesn’t change its position from the old one.
11. The error calculate will be 0 because we use the Euclidean distance to calculate the distance between the old and new centroid. When the error value is 0 the iteration stops and clustering of the data points has been done.
12. There will be convergence for sure as there are 150 instances and error value is 0 at the convergence point.
13. A plot for iris clustering has been made to show the V1 score for different k values below (instead of table) and also between SSE vs the k value.
14. The value of k has been chosen as 3 as given in problem and 7 iterations has been done to find the final centroid point.


**Part 2 – Image Clustering**

1. The dataset given is given for image clustering will be loaded into a data frame using pandas and are preprocessed with minmax scalar so that the data gets normalized from 0 to 1 so that the elements or data will have same weight.
2. There are many features used in the dataset and to reduce the data to lower dimensionality techniques like t-SNE can be used to fit the data into lower dimensionality space and PCA to remove the 0’s in the dataset. We have 784 features, so PCA has been choosed to pick the important ones and gave the V1 score as 0.69 when choosed. The score when used t- SNE is 0.59 when performed, so PCA is used. (libraries description given below)
3. Here, we initialize the centroid with 7 as the runs or iterations. Now we choose random 7 set of centroids to calculate the inner distances among them. We chose the iterations as 7 for initializing centroids as it gave us better results (different values for iter like 1-10 has been considered in the experiment). Then we calculate the distance between the old centroids and new centroid.
4. The distance used here is Euclidean distance. But we can used different approaches to calculate the distance between 2 datapoints like Euclidean distance, cosine similarity and Manhattan distance using the libraries given below in the report. Best results (score) are shown for Euclidean distance comparatively, so it has been choosed.
5. Now, with every datapoint in the dataset the distance is calculated with the centroid and the one which is nearer to the centroid is where the data or element is considered to be clustered into.
6. The new centroid value is the mean or average value of all the datapoints in the cluster formed. Then the above two steps are performed repetitively until the centroid doesn’t change its position from the old one.
7. The error calculates here is < 0.001which is the sum of square roots of the Euclidean distance between the old and the new centroid points. Here 7 runs and seed value as 25 has been choosed as they gave good results when compared to other values.
8. A plot for image clustering has been made to show the V1 score for different k values below (instead of table) and also between SSE vs the k value.
9. The value of k has been chosen as 10 as given in problem and 7 iterations has been done to find the final centroid point.

**LIBRARIES USED:**

• Python libraries like NumPy and Pandas have been used in this homework.
• NumPy supports multi-dimensional arrays that perform matrix and vector operations, and Pandas is a data manipulation tool that performs operations on rows and columns along with transforming data.
• sklearn.decomposition library has been used for performing SVM and PCA which are dimensionality reduction techniques. SVM is Support Vector Machine which is used for regression and classification. PCA is Principal Component Analysis which is used to reduce the complexity of the problem.
• sklearn.manifold library has been used for TSNE which can be used to understand high-dimensional data and fit into low - dimensional space and its used for non linear types of data where PCA works when there is linear relation in data.
• math library has been used to implement mathematic functions on data.
• sklearn.utils shuffle and random library has been used to shuffle the array or matrix (data) in a consistent way for applying clustering and random sequences are generated. It selects the random centroids calculated form the given dataset.
• matplotlib.pyplot library has been used to plot the graph to visualize the data given for clustering, plot the V1 score vs k value for iris and image clustering and also SSE vs k value in clustering (Sum of Squared errors). Along with that seaborn is a library that uses Matplotlib underneath to plot graphs.
• sklearn.metrics.pairwise library can be used to calculate the Euclidean distance and cosine similarity between 2 datapoints and also scipy.spatial.distance library can be used to find the Manhattan distance between 2 datapoints. The distances can be calculate using these inbuilt python libraries and best V1 score can be picked. In this experiment, Euclidean distance gave the best V1 score results.
• from copy import deepcopy library has been used to create a a copy of the original object into a new copy before the values in the original one changes in a recursive way.
• import warnings library has been used to ignore unnecessary warnings in the code.

<img width="628" alt="image" src="https://github.com/sanjana-govindu/Iris-and-Image-Clustering-using-KMeans-Clustering/assets/54507596/16c798f6-43d8-409e-905d-58889e854f48">

<img width="590" alt="image" src="https://github.com/sanjana-govindu/Iris-and-Image-Clustering-using-KMeans-Clustering/assets/54507596/ce0fe265-ae74-4ca7-9ba2-07c68e3dc717">


**REFERENCES USED:**
- https://towardsdatascience.com/understanding-k-means-clustering-in-machine-learning-6a6e67336aa1
- https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html 
- https://vitalflux.com/k-means-elbow-point-method-sse-inertia-plot-python/
