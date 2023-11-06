r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
**Your answer:**

1. The test set allows us to estimate our in-sample error.
No, the test set is used to estimate the out-of-sample error or the generalization error of a machine learning model.

2. Any split of the data into two disjoint subsets would constitute an equally useful train-test split.
No, if we use a tiny train set in our training-test split (for instance, one sample for training) we won't fit the model parameters as much to deal with the problem.

3. The test-set should not be used during cross-validation.
Yes, using the test set for cross validation can lead to overfitting, because our model is tuned to perform well on the given samples. We want to test the different models on unseen data to get more realistic estimation.

4. After performing cross-validation, we use the validation-set performance of each fold as a proxy for the model's generalization error.
Yes, The validation set is used during each fold to evaluate the performance of the model on a subset of the data that was not used for training.
and in the end taking the average of all the performance scores.

"""

part1_q2 = r"""
**Your answer:**

No, this approach is not justified. Adding a regularization term to the loss can help to prevent overfitting, but using the test set to pick the value of the regularization hyper parameter can lead to overfitting of the test set.

"""

# ==============
# Part 2 answers

part2_q1 = r"""
**Your answer:**

Increasing k can improve generalization of unseen data. As an example, in or model, we can see that increasing k up to k=3 improved the generalization. Increasing k further will result in an inaccurate model. This is because our estimation is influenced by further unrelated samples and their share in the total dataset.

"""

part2_q2 = r"""
**Your answer:**

1. Training on the entire train-set with various models and selecting the best model with respect to **train-set** accuracy.
Using k-fold CV we can get better estimation of the generalization of our model on unseen data, where in selecting the best model with respect to train set accuracy we can pick the model which over fit our train set the most but is unreliable for unseen data.

2. Training on the entire train-set with various models and selecting the best model with respect to **test-set** accuracy.
In k-fold CV we are using different fold in each training to validate our accuracy results.
And then taking the average or the mean of all accuracies. By doing this, our cross validation is less sensitive to train test traditional splits of data. For example, we can pick samples for testing the model which gives great results, choose the model and call it a day.

"""

# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**

Delta is defined as a hyperparameter for the model. The SVM loss is set up so that the SVM “wants” the correct class for each sample to a have a score higher than the incorrect classes by some fixed margin delta. But the weight values themselves affect the value of margin between the scores, so the value of delta in some sense is meaningless because the weights can shrink or stretch. The real tradeoff is how much we allow the weights to grow through lambda.

"""

part3_q2 = r"""
**Your answer:**

1. The linear model learns by trying to draw a hyperplane, such that the margin between the sample classes groups in space (represented as multidimensional points) will be the largest. So the learning is a learning of linear separation between regions in space. We can see an error where the digit 5 was recognized incorrectly as the digit 6.this is because the sample looks really close to that digit and may have the largest positive distance from the class linear line.

2. In KNN we're basically trying to classify the class based on division of class margins of space similar to how our linear model works, but instead of learning how to create a hyperplane that represents the margin between samples, we use the dataset itself as our estimator off the correct class by picking the K nearest neighbors and choosing the class the most of them belongs to as the decision of the correct class area the sample is in.

"""

part3_q3 = r"""
**Your answer:**

1. As can be seen, our learning rate is good, because we managed to reduce the loss fairly quickly in the first few epochs. And we don't see any resonance in the loss graph can be interpreted as taking big steps in the gradient descent algorithm which skips in each step the minimum and does not converge.

2. Our model slightly overfitting to the training set. This is because our training set accuracy is slightly better than the validation set accuracy, but the difference is small.

"""

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""
**Your answer:**

The ideal pattern to see in a residual plot is a random scatter of points with no discernible pattern, distributed randomly around zero. In the plot of the top 5 features, we can see that the rsq score is 0.68 and can be improved. By adding our non-linear features based on the existing ones and applying K-fold CV technique, we improved our model rsq score up to 0.88 on the test set. We can see the fitness of the model got improved because on our error axis at the final plot the prediction points are got closer to zero.

"""

part4_q2 = r"""
**Your answer:**

1. The linearity of our model comes from the parameters of the model, the weights and the biases to be exact. So by adding non-linear features, we increase the dimensionality of our sample space to convert the geometric shape of the problem to suite a linear line. The linear equation of the linear regression model stays linear. So the pipeline is not linear (because of the non-linearity of features) but the predictor remains linear.

2. We can fit any non-linear function of the original features with this approach. We can use a higher dimension liner regression of our addition of non-linear transformed features, with a price of further calculation.

3. Adding non-linear features will transform our space of features into a space of higher dimensions. In that hyper space, it might become easier to fit a hyperplane with decision boundary between different classes. Let's say we have a point of one class surrounded by points of different class in 2d space. By adding a new dimension of feature representing the squared Euclidean distance from the center point, we can successfully draw a linear boundary between the two classes. The new hyperplane is still a linear hyperplane in the new hyperspace, but would not remain a linear hyperplane in the old feature space.

"""

part4_q3 = r"""
**Your answer:**

1. Lambda in our case controls the magnitude of the regularization term. The log scale is used where small changes of hyperparameter value can have dramatic effect on the model performance. Log scale allows us to explore various range of hyperparameter values that a very by several orders of magnitude. This logarithmic scale is suitable for those kinds of hyperparameters to deal with their exponential relationship nature.

2. We are checking 20 lambda values,3 degree values and using 3 folds. This gives us 180 iterations of fitting the model to the data.

"""

# ==============
