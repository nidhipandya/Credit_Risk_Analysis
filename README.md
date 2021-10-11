## Credit_Risk_Analysis

### Overview of the analysis
This analysis is done to predict credit risk using supervised machine learning. Credit risk is an inherently unbalanced classification problem, as good loans easily outnumber risky loans. So, I used different techniques to train and evaluate models with unbalanced classes. here I used imbalanced-learn and scikit-learn libraries to build and evaluate models using resampling. 

I used the oversample method on the data using the RandomOverSampler and SMOTE algorithms, and undersample method on the data using the ClusterCentroids algorithm. Then, I used a combinatorial approach of over- and undersampling using the SMOTEENN algorithm. Next, I compared two new machine learning models that reduce bias, BalancedRandomForestClassifier and EasyEnsembleClassifier, to predict credit risk. 

### Results

Here are three primary measures are considered while testing performace of these modals: accuracy, precision, and recall.

* **Accuracy** is the ratio of correct predictions to the total number of input samples
* **f1** score is a simplified measure of model performance. It is a weighted harmonic mean of precision and recall
	* **Precision** is the ability of a classifier not to label an instance positive that is actually negative
	* **Recall** is the ability of a model to find all positive instances

Below are the results of different algorithms

#### Oversampling

* Naive Random Oversampling

Random oversampling duplicates examples from the minority class in the training dataset and can result in overfitting for some models. Also here, instances of the minority class are randomly selected and added to the training set until the majority and minority classes are balanced.

![img1](https://github.com/nidhipandya/Credit_Risk_Analysis/blob/main/Resources/images/img1.PNG)

The accuracy score of this model is around 64%
The precision for low risk loans is almost 100% but the precision for high risk loans is very low around 1%.

* SMOTE Oversampling

SMOTE is an oversampling technique that generates synthetic samples from the minority class.

![img2](https://github.com/nidhipandya/Credit_Risk_Analysis/blob/main/Resources/images/img2.PNG)

The accuracy score of this model is around 62%
The precision for low risk loans is almost 100% but the precision for high risk loans is very low around 1%.

#### Undersampling

* Cluster centroid undersampling

Undersample by generating centroids based on clustering methods. The algorithm identifies clusters of the majority class, then generates synthetic data points, called centroids, that are representative of the clusters.

![img3](https://github.com/nidhipandya/Credit_Risk_Analysis/blob/main/Resources/images/img3.PNG)

The accuracy score of this model is around 51%.
The precision for low risk loans is almost 100% but the precision for high risk loans is very low around 1%.

#### Combination (Over and Under) sampling

* Combination sampling

SMOTEENN combines the SMOTE and Edited Nearest Neighbors (ENN) algorithms. SMOTEENN is a two-step process:

	1. Oversample the minority class with SMOTE.
	2. Clean the resulting data with an undersampling strategy. If the two nearest neighbors of a data point belong to two different classes, that data point is dropped.

![img4](https://github.com/nidhipandya/Credit_Risk_Analysis/blob/main/Resources/images/img4.PNG)

The accuracy score of this model is around 64%.
The precision for low risk loans is almost 100% but the precision for high risk loans is very low around 1%.

#### Balanced Random Forest classifier

* Balanced Random Forest Classifier

A balanced random forest randomly under-samples each boostrap sample.

![img5](https://github.com/nidhipandya/Credit_Risk_Analysis/blob/main/Resources/images/img5.PNG)

The accuracy score of this model is around 87%.
The precision for low risk loans is almost 100% but the precision for high risk loans is very low around 3%. here precision is increased compared to other models.

#### Easy Ensemble Adaboost 

* Easy Ensemble AdaBoost Classifier

Ensemble learning is the process of combining multiple models, like decision tree algorithms, to help improve the accuracy and robustness, as well as decrease variance of the model

![img6](https://github.com/nidhipandya/Credit_Risk_Analysis/blob/main/Resources/images/img6.PNG)

The accuracy score of this model is around 93%.
The precision for low risk loans is almost 100% but the precision for high risk loans is very low around 9%. It has highest precision for high risk loans compared to other models.

### Summary
The results are almost same for oversampling and undersampling models. But the results highly improve with the Balanced Random Forest classifier and Easy Ensemble Adaboost Classifier. The accuracy score is around 90%.

##### Recommendation
EasyEnsembleClassifier model can identify 93% of risky loans also It has a precision of almost 100% for good loans and only 9 % for high risk loans, looking at the confusion matrix, there are lot of false negatives and it failed to notice several good loan applications. but compared to others, this model is better to use.

Processing time should also be taken into consideration as the models all run at different rates of speed - this factor will be important when processing on a large-scale and considering service level expectations.
