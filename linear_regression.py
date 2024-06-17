import numpy as np

def compute_Phi(x, p):
    '''
    Compute the feature matrix Phi of x. We will construct p polynomials, the p features of the data samples.
    The features of each sample are x^0, x^1, x^2, ..., x^(p-1).
    Input:
        x: a vector of samples in one dimensional space, a numpy vector of shape (n,).
           Here n is the number of samples.
        p: the number of polynomials/features
    Output:
        Phi: the design/feature matrix of x, a numpy array of shape (n,p).
    '''
    Phi = np.vstack([x**i for i in range(p)]).T
    return Phi

def compute_yhat(Phi, w):
    '''
    Compute the linear logit value (predicted value) of all data instances. z = <w, x>
    Here <w, x> represents the dot product of the two vectors.
    Input:
        Phi: the feature matrix of all data instances, a float numpy array of shape (n,p).
        w: the weights parameter of the linear model, a float numpy array of shape (p,).
    Output:
        yhat: the logit value (predicted value) of all instances, a float numpy array of shape (n,)
    '''
    yhat = np.dot(Phi, w)
    return yhat

def compute_L(yhat, y):
    '''
    Compute the loss function: mean squared error divided by 2. In this function, divide the original mean squared error by 2 for making gradient computation simpler.
    Input:
        yhat: the predicted sample labels, a numpy vector of shape (n,).
        y: the sample labels, a numpy vector of shape (n,).
    Output:
        L: the loss value of linear regression, a float scalar.
    '''
    L = np.mean((yhat - y) ** 2) / 2
    return L

def compute_dL_dw(y, yhat, Phi):
    '''
    Compute the gradients of the loss function L with respect to (w.r.t.) the weights w.
    Input:
        Phi: the feature matrix of all data instances, a float numpy array of shape (n,p).
        y: the sample labels, a numpy vector of shape (n,).
        yhat: the predicted sample labels, a numpy vector of shape (n,).
    Output:
        dL_dw: the gradients of the loss function L with respect to the weights w, a numpy float array of shape (p,).
    '''
    dL_dw = np.dot(Phi.T, (yhat - y)) / len(y)
    return dL_dw

def update_w(w, dL_dw, alpha=0.001):
    '''
    Update the weight vector using gradient descent.
    Input:
        w: the current value of the weight vector, a numpy float array of shape (p,).
        dL_dw: the gradient of the loss function w.r.t. the weight vector, a numpy float array of shape (p,).
        alpha: the step-size parameter of gradient descent, a float scalar.
    Output:
        w: the updated weight vector, a numpy float array of shape (p,).
    '''
    w -= alpha * dL_dw
    return w

def train(X, Y, alpha=0.001, n_epoch=100):
    '''
    Train the linear regression model by iteratively updating the weights w using gradient descent.
    We repeat n_epoch passes over all the training instances.
    Input:
        X: the feature matrix of training instances, a float numpy array of shape (n, p). Here n is the number of data instance in the training set, p is the number of features/dimensions.
        Y: the labels of training instance, a numpy integer array of shape (n,).
        alpha: the step-size parameter of gradient descent, a float scalar.
        n_epoch: the number of passes to go through the training set, an integer scalar.
    Output:
        w: the weight vector trained on the training set, a numpy float array of shape (p,).
    '''
    p = X.shape[1]  # Number of features
    w = np.zeros(p)  # Initialize weights as zero
    for _ in range(n_epoch):
        yhat = compute_yhat(X, w)  # Compute predictions
        dL_dw = compute_dL_dw(Y, yhat, X)  # Compute gradients
        w = update_w(w, dL_dw, alpha)  # Update weights
    return w
