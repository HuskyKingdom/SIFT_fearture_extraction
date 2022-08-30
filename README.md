

<br />
<div align="center" id="readme-top">
  
 <br />
<img src="https://raw.githubusercontent.com/HuskyKingdom/Grasp_Detection/main/imgs/2.png" width="300" height="300"></br>
</br>
  <h1 align="center">SIFT Feature Extraction Algorithm</h1>

  <p align="center" >
  Distributed under the MIT License.

This project involves the approach to extract SIFT(Scale-Invariant Feature Transform) features from training data of given dataset with 5 different classes, and clustering the result SIFT descriptors to form an dictionary of Bag-of-words representation. Eventually, generate keywords histograms for both training data and testing data, as well as label them to one of the cluster centers, classify the testing images to its nearest neighbor according to the distances of histograms.
After classification of testing images, the program will run a set of tests to evaluate the classification performance. Moreover, the program also demonstrates the classification progress using varies histogram comparing mechanisms or different clustering parameters.

<br />
<a href="https://yuhang.topsoftint.com">Contact me at: <strong>yuhang@topsoftint.com</strong></a>

<a href="https://yuhang.topsoftint.com"><strong>View my full bio.</strong></a>
    <br />
    <br />
  </p>
</div>



<!-- ABOUT THE PROJECT -->
## Instructions

**Note that, since the boundary checking is not implemented, each option/index entered much be from given options or index that within meaningful range.**

The project files are with the following directories:
- SIFT.py
- cluster
- ./features
- ./COMP338_Assignment1_Dataset

Where the SIFT is the project code, cluster is the saved clustered centers variable list of type sklearn.kmeans object. Two folders, features and COMP338_Assignment_Dataset are the folder for saved SIFT features, and given image dataset, respectively.

1. Feature Extraction of Training data.

Run the python file SIFT.py.

The program first reading the images from the given dataset in the corresponding directories, then shows the welcoming message with two options to perform the SIFT feature extraction on training data.
Usually, by our testing, the average time taken for the extraction of images will take up to one hour for the given 350 training images, this is however not convenient for code testing during the programing stage, therefore we made the program to be able to save the features ever time it extracted them, to the feature directory, as well as load the saved files when needed. The following list shows the SIFT feature contents saved, with respect to the file name in feature directory:

_“SIFT_Features_Train”_ – SIFT keypoint descriptors from training images 
_“SIFT_Features_Train_Points”_ - SIFT keypoints from training images

It is recommended to enter option 2 to directly load the extracted features for time saving purpose. The technical details of SIFT feature extraction is explained in later section.

2. Dictionary Generation

Once successfully obtained the features from training dataset, the program will then cluster the SIFT descriptors of training data to any numbers of cluster centers. The number of centers is given by the variable Num_WORDS at line 14.

Note that this step is time consuming too by our testing, therefore the program also gives an option to load the clustered centers instead of performing clustering.

If letter “Y” is entered, the program will load the cluster centers from file “cluster”. Otherwise, if the letter “N” is entered, the program will perform k-mean clustering to all descriptors of training dataset and save the resulting clustering center list.

3. Image representation with a histogram of codewords

With the clustering centers ready, the program then ready to represent images from both training and testing dataset as histograms of clustered centers.

Firstly, the program will read the testing images, and then try to extract the features from them as we did to training dataset. The loading per-trained features option and perform feature extraction option are both given, as shown below:

The following list shows the SIFT feature contents saved, with respect to the file name in feature directory:

_“SIFT_Features_Test”_ – SIFT keypoint descriptors from testing 
images 
_“SIFT_Features_Test_Points”_ - SIFT keypoints from testing images

It is recommended to enter option 2 to directly load the extracted features for time saving purpose.
After obtaining SIFT features of testing images, we now need to label each descriptor from both training and testing images into one of the clustered centers.

It is recommended to enter option 1 to directly load the labels for time saving purpose. Please also note that the labelling is using L2 distance between descriptors, this will be explained in detail in later section.
Once the labeling is done, the program then allows you to view some image patches from a certain class label.

Please note that index entered must not exceed (the number of clustered centers - 1), otherwise the program will exist with errors, since the boundary checking is not required.

Please see an example shows following with class label “5” is entered:

<img src="https://raw.githubusercontent.com/HuskyKingdom/Grasp_Detection/main/imgs/3.png" width="300" height="300">

Note that if an descriptor and its corresponding keypoint is at the location that on the edge of the image and with high sigma value, the image patch of it might not be drawn correctly since the information is too abstract.

You can enter -1 to continue or enter another class label to view image patches, as suggested by the program output.

Then, if -1 is entered, the program will generate the BOW(Bag-of-Words) histograms for each image in both training and testing dataset. Implementation Details will be explained in later section.

Please not that, for testing purpose, the program will also store generated labels and histograms into “./features” folder for both datasets.

The following list shows the meanings of each file saved, with respect to the file name in feature directory:

“Traing_labels” – Saved labels for each descriptor of training images with respect to cluster centers

“Test_labels” – Saved labels for each descriptor of testing images with respect to cluster centers

“Traing_hist” – Saved BOW histogram of training dataset

“Test_hist” - Saved BOW histogram of testing dataset

4. Classification & Evaluation

After obtained the BOW histograms of each image, then program then performs L2 distance on each BOW testing image histograms to compare to all BOW training image histograms, and hence conclude the corresponding testing image to the same image class as a certain BOW training image histogram, which has minimum L2 BOW histogram distance to the current testing BOW histogram.
By comparing the grand truth of all testing images and their conclude(classified) classes, the program evaluate the performance of the classification and showing the error rates.

You can also view the images that and correctly & incorrectly classified in each 0-4 different class, where the following list shows the relation between class indexes and class name:

- Class 0 – Airplanes
- Class 1 – Cars
- Class 2 – Dog
- Class 3 – Faces
- Class 4 – Keyboard
- Enter -1 to skip this stage.

5. Histogram Intersection

By replacing the L2 method with the histogram intersection method for comparing histograms, the program runs step 4 and 5 again, and present the result.The classifying result viewing and confusion matrix generation is also performed in this particular step.

6. Reducing the number of clustered centers

As mentioned in the documentation of this assignment, in step 7 we need to perform an experiment with different number of clustering centers.

In my implementation, this is achievable by simply change the variable Num_WORDS in line 14 of the code. This value is set by default to 500.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



## Implementation Details & Theory

Provided <a href="">here</a>. Please do not directly copy.



## References / Related Works
<p id="6"></p>

GitHub resources[1]:
https://github.com/rmislam/PythonSIFT

David G. Lowe`s Paper[2]:
https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf

External library[3]:
https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

Keypoint obj by OpenCV:
https://docs.opencv.org/3.4/d2/d29/classcv_1_1KeyPoint.html

Taylor series:
https://en.wikipedia.org/wiki/Taylor_series

K-Means Algorithm & KNN:
https://en.wikipedia.org/wiki/K-means_clustering 

https://towardsdatasciencecoma-simple-introduction-to-k-nearest-neighbors-algorithm-b3519ed98e


<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- LICENSE -->
## License

Distributed under the MIT License.

<p align="right">(<a href="#readme-top">back to top</a>)</p>




