r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 (Backprop) answers

part1_q1 = r"""
**Your answer:**
1.The matrix \textbf{X} is of the shape 64x1024 and \textbf{W} is of shape of 512x1024 so according to our linear layer $\displaystyle \mathbf{Y} =\mathbf{XW}^{T}$ is of shape 64x512 and can be written as a sum of $\displaystyle \mathbf{Y}_{jk} =\sum \mathbf{X}_{ji}\mathbf{W}_{ik}^{T}$

so $\displaystyle \left(\frac{\partial \mathbf{Y}}{\partial \mathbf{X}}\right)_{j,k,m,n} =\frac{\partial Y_{jk}}{\partial X_{m}{}_{n}} =\frac{\partial }{\partial X_{m}{}_{n}}\sum X_{ji} W_{ik}^{T} =\sum \frac{\mathbf{\partial }}{\partial \mathbf{X}_{m}{}_{n}} X_{ji} W_{ik}^{T} =\delta _{jm} \delta _{in} W_{ik}^{T} =\delta _{jm} W_{kn}$

A. the tensor is a 4D tensor of shape 64x512x64x1024

B. as we can see from our calculation we have a kroniker delta which zero out the element where

the row of the element Y is not the same as the row of the element X. that is why the Jacobian is sparse.

C. No we don't, by using the chain rule technique we can write
\begin{equation*}
\frac{\partial L}{\partial \mathbf{X}} =\frac{\partial L}{\partial \mathbf{Y}}\frac{\partial \mathbf{Y}}{\partial \mathbf{X}} =\sum _{j,k} ,\frac{\partial L}{\partial Y_{j}{}_{k}}\frac{\partial Y_{j}{}_{k}}{\partial X_{m}{}_{n}} =\sum _{j,k}\frac{\partial L}{\partial Y_{j}{}_{k}} \delta _{jm} W_{kn} =\sum _{k}\frac{\partial L}{\partial Y_{mk}} W_{kn} =\delta \mathbf{Y} \cdot \mathbf{W}
\end{equation*}
So we only need to multiply by $\displaystyle \mathbf{W}$

2. Let's repeat the calculation for $\displaystyle \frac{\partial \mathbf{Y}}{\partial \mathbf{W}}$:
\begin{equation*}
\left(\frac{\partial \mathbf{Y}}{\partial \mathbf{W}}\right)_{j,k,m,n} =\frac{\partial Y_{jk}}{\partial W_{m}{}_{n}} =\frac{\partial }{\partial W_{m}{}_{n}}\sum X_{ji} W_{ik}^{T} =\sum X_{ji}\frac{\partial }{\partial W_{nm}^{T}} W_{ik}^{T} =\delta _{in} \delta _{km} X_{ji} =\delta _{km} X_{jn}
\end{equation*}
A. the tensor is a 4D tensor of shape 64x512x512x1024

B. as we can see from our calculation we have a kroniker delta which zero out the element where

the column of the element Y is not the same as the row of the element W. that is why the Jacobian is sparse.

C. No we don't, by using the chain rule technique we can write
\begin{equation*}
\frac{\partial L}{\partial \mathbf{W}} =\frac{\partial L}{\partial \mathbf{Y}}\frac{\partial \mathbf{Y}}{\partial \mathbf{W}} =\sum _{j,k} ,\frac{\partial L}{\partial Y_{j}{}_{k}}\frac{\partial Y_{j}{}_{k}}{\partial W_{m}{}_{n}} =\sum _{j,k}\frac{\partial L}{\partial Y_{j}{}_{k}} \delta _{km} X_{jn} =\sum _{j} X_{nj}^{T}\frac{\partial L}{\partial Y_{j}{}_{m}} =\mathbf{X}^{T} \cdot \delta \mathbf{Y}
\end{equation*}
So we only need to multiply by $\displaystyle \mathbf{X}^{T}$
"""

part1_q2 = r"""
**Your answer:**
2. No, we can use other techniques as we learned in the tutorial to train neural networks with descent based optimization. For example, Forward/Reverse mode AD or even calculating the gradients by hand specific to our model (and using the equation of the gradient after each forward phase) or even using the likelihood ratio (LR) method (https://arxiv.org/abs/2305.08960).the reason why backpropagation is so popular technique is that it allows us to use the power of dynamic programming to calculate gradients efficiently with computational graphs.
"""


# ==============
# Part 2 (Optimization) answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    wstd, lr, reg = 0.1, 0.05, 0
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = (
        0,
        0,
        0,
        0,
        0,
    )

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = (
        0.1,
        0.05,
        3e-3,
        14e-5,
        0
    )
    # ========================
    return dict(
        wstd=wstd,
        lr_vanilla=lr_vanilla,
        lr_momentum=lr_momentum,
        lr_rmsprop=lr_rmsprop,
        reg=reg,
    )


def part2_dropout_hp():
    wstd, lr, = (
        0,
        0,
    )
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    wstd, lr, = (
        0.1,
        1e-3
    )
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Your answer:**
1. Yes, our graph results match our expectations. Dropout layers prevent from our model to overfit

to the trained data. We can see in our graph that the model that gets the best train accuracy is the one which has not dropout layers activated but not so much not so much on the test set. So, other than acting as kind of a regularization layer, it also prevents from the network to depending on dominant features and thus allows for the network to generalize even more. When we apply too much Dropout, we prevent from the backpropagation algorithm to fine tune weights at the same speed as a non dropout network and produce under fitting and loss of expressiveness. In our results, the Dropout of 0.8 suffers from bad results in train and test accuracy. Furthermore, applying too much Dropout can make the network more sensitive to changes of hyperparameters. The model that don't overfit and at the same time gives the best generalization is the model with dropout of 0.4
"""

part2_q2 = r"""
**Your answer:**
Yes, the accuracy is basically a discrete technique to measure how well our model fits the given dataset, but the cross entropy measures how confident the model is about the classification of our input. So for cross entropy loss function
\begin{equation*}
L_{CE} =-\sum _{i-1}^{n} t_{i} \cdotp log( p_{i}) \ for\ n\ classes
\end{equation*}
The $\displaystyle p_{i}$ can decrease for many "1" binary class because of changes of model parameters, and at the same time, more correct predictions occur because of a change of delta in the probability passing the threshold and increasing the accuracy even though the model is less confident. For example, for binary classification a change from [0.9, 0.1] to [0.6, 0.4] outputs produce the same classification which won't change the accuracy, but the loss increased from 0.37 to 0.6 while but small change in other sample from  [0.45, 0.55] to  [0.55, 0.45] won't change the sum of all losses (given the other change remains) in the cross entropy but will make the accuracy increase.
"""

part2_q3 = r"""
**Your answer:**
3.
1. backpropagation is used to calculate the model parameters by doing a forward pass and using saved data for propagating backward up to the input layer and getting the gradients by chain rule multiplications.
Gradient descent uses the  given gradient to make a step or do a small change in the model parameters (weight and biases) to reduce the error on the given cost function in a hope to make the model suitable for prediction on unseen data.
2. in GD the gradients are computed using the entire training data, while in SGD we first divide the training data into samples or mini batches. Then in each iteration, we pick one sample and make a gradient step based on it. Because in SGD we don't use the entire dataset, SGD is reducing the computation burden. The second thing is that the parameters are frequently updates relative to GD which converges slowly, and it is more stable but not used so often for large datasets
3. in deep learning, we are most of the time are provided with huge datasets. Doing GD on the entire dataset is computationally slower and requires huge amounts of memory to hold all the results, and it is prone to numerical errors due to large amount of algebraic operations. That is why SGD is more popular in deep learning because this technique solves those problems by using mini batches of samples to calculate the gradients, which helps us to train the network quicker and with less numerical errors and not using all the memory available to us.
4.
A. We can use this approach only when our cost function is linear. Let's say we use an empirical risk function $\displaystyle C=\sum _{i} L( h( x_{i}) ,y_{i}) =\sum _{i} L_{( i)}$ and define $\displaystyle C_{( i)}$ which compose of mini batch to calculate the loss with respect to this batch:
\begin{equation*}
\sum _{i=1}^{n}\frac{\partial }{\partial \mathbf{\theta }} L_{( i)} =\frac{\partial }{\partial \mathbf{\theta }}\sum _{i=1}^{n_{1}} L_{( i)} +...+\frac{\partial }{\partial \mathbf{\theta }}\sum _{i=n_{m-1} +1}^{n_{m}} L_{( i)} =\frac{\partial }{\partial \mathbf{\theta }} C_{( 1)} +...+\frac{\partial }{\partial \mathbf{\theta }} C_{( m)} =\frac{\partial }{\partial \mathbf{\theta }}( C_{( 1)} +...+C_{( m)}) =\frac{\partial }{\partial \mathbf{\theta }} C
\end{equation*}
So, as we can see, there are different ways to calculate the cost only if we use linear cost function

B. For each batch, we need to store the result of the forward pass. Next, we put together all the relevant data for each batch in memory in order to make a backward pass on it, so we get a memory exhaustion.
"""

part2_q4 = r"""
**Your answer:**
4. 

1. We can do a classic forward mode AD $\displaystyle v_{j+1} .grad\leftarrow v_{j+1} .fn.derivative( v_{j} .val) \cdotp v_{j} .grad$

to reduce the memory complexity to $\displaystyle O( 1)$ we will use a single variable for grad calculation and single variable for input from last node, so the equation can be reduced to
\begin{equation*}
Grad\leftarrow v_{j+1} .fn.derivative( Input) \cdotp Grad
\end{equation*}
where Grad=1 and Input=$\displaystyle v_{0} .val$

2. We can do a classic backward mode AD $\displaystyle v_{j-1} .grad\leftarrow v_{j} .fn.derivative( v_{j-1} .val) \cdotp v_{j} .grad$

to reduce the memory complexity to $\displaystyle O( 1)$ we will use a single variable for grad calculation and single variable for input from last node, so the equation can be reduced to
\begin{equation*}
Grad\leftarrow v_{j} .fn.derivative( Input) \cdotp Grad
\end{equation*}
where Grad=1 and Input=$\displaystyle F_{n}( x)$

At first thought for a general computational graph with multiple input nodes, we will need 2 variables
for each input node. So we get a $\displaystyle O( n)$ memory complexity, but we can instead traverse from input to output node path for each gradient element, lowering the memory complexity to $\displaystyle O( 1)$. The problem is that this cannot be used when trying to make parallel computation without going back to 2n variables.

3. this technique helps us to fine tune minimal amount of parameters for our model to see their effects on the model performance without using large amounts of memory as we saw. The downside is that we cannot parallel this process for parameters at the same time without using $\displaystyle O( n)$ memory.
"""

# ==============


# ==============
# Part 3 (MLP) answers


def part3_arch_hp():
    n_layers = 0  # number of layers (not including output)
    hidden_dims = 0  # number of output dimensions for each hidden layer
    activation = "none"  # activation function to apply after each hidden layer
    out_activation = "none"  # activation function to apply at the output layer
    # TODO: Tweak the MLP architecture hyperparameters.
    # ====== YOUR CODE: ======
    n_layers = 2  # number of layers (not including output)
    hidden_dims = 500  # number of output dimensions for each hidden layer
    activation = "relu"  # activation function to apply after each hidden layer
    out_activation = "none"  # activation function to apply at the output layer
    # ========================
    return dict(
        n_layers=n_layers,
        hidden_dims=hidden_dims,
        activation=activation,
        out_activation=out_activation,
    )


def part3_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    loss_fn = torch.nn.CrossEntropyLoss()  # One of the torch.nn losses
    lr, weight_decay, momentum = 0.5, 3e-3, 0.1  # Arguments for SGD optimizer
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part3_q1 = r"""
**Your answer:**
1.

1. We have a relatively low Optimization error. As can be seen from the graphs, when training the model we reach a relatively low training loss and keep it steady throughout the training. At the same time, our accuracy reaches a high percentage rate. From it, we can conclude that the model can fit well to the given training data.

2. Our Generalization error is higher than the Optimization error, and experience jumps during the training in both the test loss and test accuracy. But what we can see is an overall improvement in the jump as we keep training, but it doesn't go a way, so over all we are having high Generalization error.

3. If we look at the decision boundary we can see that the model succeeded in drawing an accurate boundary between the different classes and therefore the approximation error is low.
"""

part3_q2 = r"""
**Your answer:**
2. Based on the confusion matrix we derived for the training set, we can see that the FNR is greater than the FPR, we could expect the model to do the same on the validation set. But when we look at the decision boundary graph of the validation data set, there are a relatively high number of samples of class 1 in the area of class 0, so the model would wrongly predict those samples as of class 1. So from this point of view, we actually predict that the model will have a bigger FPR than FNR in the validation set, as we got from the confusion matrix.
"""

part3_q3 = r"""
**Your answer:**
3.
1. For a person with a disease that will develop non-lethal symptoms, a high false positive rate can be riskier (assume that these further tests are expensive and involve high-risk to the patient) than the chances of false detecting the illness of a sick person. Therefore, FPR is expensive than FNR 
and we will pick the optimal point on the ROC curve to minimize the FPR.
2. In this scenario, wrongly diagnosing a patient as healthy even though he is sick is more dangerous than the case of getting a false positive. Therefore, FNR is expensive than FPR 
and we will pick the optimal point on the ROC curve to minimize the FNR.
"""

part3_q4 = r"""
**Your answer:**
4.
1. As we can see, the test and validation accuracy improves as we go down the column of the graph table. By increasing the width of the model it becomes more expressive and by that produces a more "breakable" decision line.
2. When we travel in the rows, we see that the model is getting improved when we increase the model depth to depth=2. This can be explained because each layer adds another activation function acting on a previously modified feature vector of each given sample, which increases the non-linearity capabilities of the network to draw a curved boundary decision line. When we increased the depth to 4, our model started to overfit the training data, and we have experienced a minor decrease in our test accuracy results.
3. We can see from our results that a network that consists of one layer with 32 neurons gives better validation accuracy and test accuracy than the network that consists of 4 layers with width of 8. This can be explained by the general guidance principle that wider networks can capture complex patterns in the given data while deeper networks can extract more abstract patterns from the data which is not our main desire from the given task.
4. The threshhold selection given the validation set did not improve the test accuracy result. Because the samples distribute differently on both sets, picking the right threshold for one dataset doesn't correlate with the improvement of the model on the other set.
"""
# ==============
# Part 4 (CNN) answers


def part4_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    loss_fn = torch.nn.CrossEntropyLoss()  # One of the torch.nn losses
    lr, weight_decay, momentum = 0.03, 2e-3, 2e-3  # Arguments for SGD optimizer
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part4_q1 = r"""
**Your answer:**
1.

1. for the first conv we get $\displaystyle 256\cdot ( 256\cdot 3\cdot 3+1) =590080$ parameters (including biases) and a second conv that gives us another 590080. So summing up the number of parameters of each conv layer, and we get 1180160 parameters. For the bottleneck block, we have a total number of:
\begin{equation*}
64\cdot ( 1\cdot 1\cdot 256+1) +64\cdot ( 3\cdot 3\cdot 64+1) +256\cdot ( 1\cdot 1\cdot 64+1) =70016
\end{equation*}
parameters.

2. The equation to calculate the number of flops produced by a given conv layer is 
\begin{equation*}
\ FLOPs\ =\ (( K_{h} \cdot K_{w}) \cdot C_{in} +1) \cdot (( H_{out} \cdot W_{out}) \cdot C_{out})
\end{equation*}
because we are having smaller kernel sizes of conv layers in the bottleneck block, we are performing higher number of FLOPs in the regular block than the bottleneck residual block.

3. 
1. The bottleneck residual block has 2 1x1 conv layers used to reduce the spatial dimensions of the feature map and restore the number of channels at the end of the block, while the regular block preserves the spatial dimensions of the feature map. That is why bottleneck blocks are helpful in doing computationally expensive 3x3 conv. Bottleneck residual blocks can potentially have different output feature map dimensions compared to the input feature map dimensions.

2. A bottleneck residual block in a ResNet architecture is designed to capture and represent more features compared to a regular block.
"""

# ==============

# ==============
# Part 5 (CNN Experiments) answers


part5_q1 = r"""
**Your answer:**
1.
1. When we add more depth to the CNN we are increasing the capacity of the forward chain to capture complex patterns in the data and the ability of the network to derive an abstract representation of the input data. We can see from our results that 4 layers got us the best test accuracy, but at the cost of fitting the test data too much. One of the reasons for not keeping this up with more depth is the amount of training time for a CNN with large amount of layers to get the network parameters to sufficient local minima in higher dimensionality space. The thing is that event though we didn't get the same train loss at depth of 16 as the other depths, our test lost was even better, suggesting we had better generalization capacity with a network of large depth.
2. Yes, it happened for L=16 with K=64, and it might be the problem of vanishing gradients which can occur when the derivatives calculated during the backpropagation phase are smaller than one and so multiplying them by the chain rule technique gives tiny gradients which make the learning step of updating the model parameters identical to the previous iteration and thus keeps the network in the same state. This can be solved by using different architectures like the Resnet to introduce residual connections which allow the gradients to pass multiple layers by identity mappings or use batch normalization which normalize each layer input by recentering and rescaling it.
"""

part5_q2 = r"""
**Your answer:**
2. At L=2 We have CNN with K=128 which performs badly on train loss and train/test accuracy
while L=2 with K=32 and K=64 gives the same results. When we increase the depth to 4, all the CNNs start to overfit the training set, as all CNN reaches train accuracy of roughly 100% while getting test accuracy result below 70%. At L=8 we see improvements in our steadiness of our test accuracy results, while the result are similar to the previous L=4 experiment. On both L=4 and L=8 K=32 gives poor test accuracy results. We can learn from it that 64 filters for us is a sweet spot for both large depth and small depth for our learning task. We want a high number of filters per layer because it allows us to more diverse range of visual patterns, potentially improving its ability to detect and represent various features, but we have to make sure as experiment 1.1 showed that we also need to increase the depth to make sure our network can make higher abstraction from all the extracted features and generalize better.
"""

part5_q3 = r"""
**Your answer:**
3. experiment 1.3 shows us that all the models get roughly the same training accuracy on all different depth. While depth of 2 converges quicker than the other 2 depths types. The explanation for that is the reduction of the number of multiplication the trainer required to do to get the gradients, and by that making them bigger in each step. With the change of adding different filter channels per block, we increase the diversity of each filter activation layers of the feature map. So our CNN improved even further, especially with CNN with depth of 3.
"""

part5_q4 = r"""
**Your answer:**
4. When we keep the same number of filters per conv layer, the model which succeeds the best in generalization is a CNN of depth of 32 with K=32. The interesting part is that the model train loss score is lower than the score of the other models and the slope absolute value is bigger for the others for obvious reasons. This can be reflected on the train accuracy as well, but when we look at the test loss scores, the L=32 k=32 model performs dramatically better than the other model and produces accuracy of 75\%-80\% on the test set while as we decreased the number of depth the models produce worse accuracies on the test set. When we look at the other experiment, we have very similar behavior. Network with greater depth tends to perform better on test set both accuracy and test loss but instead of improvement on test scores relative to fixed K parameter the different channel filters are lowering the score. Happen not because of the increase of channels of feature maps but instead by the decrease of depth, so the general rule we can learn from it that to better use the generated feature maps of each residual block we have to use more depth to abstract the features better. Compared to experiment 1.3 and 1.1 residual blocks help us in both vanishing gradients problems (because of the shortcut connections) but even help to keep early learned features close to the mlp part of the network in the forward sense. 
"""

part5_q5 = r"""
**Your answer:**
4. When we keep the same number of filters per conv layer, the model which succeeds the best in generalization is a CNN of depth of 32 with K=32. The interesting part is that the model train loss score is lower than the score of the other models and the slope absolute value is bigger for the others for obvious reasons. This can be reflected on the train accuracy as well, but when we look at the test loss scores, the L=32 k=32 model performs dramatically better than the other model and produces accuracy of 75\%-80\% on the test set while as we decreased the number of depth the models produce worse accuracies on the test set. When we look at the other experiment, we have very similar behavior. Network with greater depth tends to perform better on test set both accuracy and test loss but instead of improvement on test scores relative to fixed K parameter the different channel filters are lowering the score. Happen not because of the increase of channels of feature maps but instead by the decrease of depth, so the general rule we can learn from it that to better use the generated feature maps of each residual block we have to use more depth to abstract the features better. Compared to experiment 1.3 and 1.1 residual blocks help us in both vanishing gradients problems (because of the shortcut connections) but even help to keep early learned features close to the mlp part of the network in the forward sense. 
"""

# ==============

# ==============
# Part 6 (YOLO) answers


part6_q1 = r"""
**Your answer:**
1. 

1. The model didn't detect the objects in the pictures very well. For the first image, the model detected two dolphins as a person and one dolphin as a surfboard. For the second image, two of the three dogs were recognized as cats, and the cat in the picture did not get recognized at all.

Even though the model is not good at classification, the regression capabilities of the model on the given 2 pictures is better in a way for most objects it successfully matches a bounding box for their size in the pictures.

2. \ Yolov5s is a smaller and lighter version of the YOLO model compared to its larger counterparts. With a smaller model size, Yolov5s may have limited capacity to capture complex patterns and details in the input data. The other possible reason is that the model training set was not diverse. The performance of any object detection model, including Yolov5s, heavily relies on the quality and diversity of the training dataset.
"""


part6_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part6_q3 = r"""
**Your answer:**
3. We picked 3 different kinds of setups for object pitfalls: a blurred picture of a cat, a low light image of a person and a picture of a cluttered crowd of people. For the blurred cat picture, the model detected the cat's eyes as sports ball, which can seem like a tennis balls. The model didn't succeed in classifying the cat in the picture or even detect the object. For the dark image of the man, we had no success in detection of the object and for the picture of the crowd, the model detected a blurred line in the picture and classified it as a baseball bat, ignoring all the people in the picture. For the blurring effect, the reason for the bad performance is that the blurring deforms important features from the picture necessary for the model to figure out that there is a cat while making some cat's features like the eye similar to features of another class like a sport ball. The light illumination hides other features relevant for person localization and classification, like the shape of the head, leaving a small subset of features for the model to classify. The cluttering presents numerous distracting or similar-looking objects that add a lot of noise (especially by wearing hats and all kinds of accessories) and make the model struggle to differentiate between all the target objects.
"""
