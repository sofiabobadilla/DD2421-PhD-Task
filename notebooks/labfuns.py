from __future__ import absolute_import, division, print_function
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from numpy import genfromtxt
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Ellipse
from matplotlib.colors import ColorConverter
import random as rnd
from sklearn.datasets import make_blobs
from sklearn import decomposition, tree
from scipy import misc
from importlib import reload
import warnings
warnings.filterwarnings('ignore')

# import seaborn as sns
# sns.set()

def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the
    ellipse patch artist.

    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)
    ellip.set_alpha(0.25)

    ax.add_artist(ellip)
    return ellip


# Splits data into training and test set, pcSplit defines the percent of
# the data should be used as training data.
def trteSplit(X,y,pcSplit,seed=None):
    # Compute split indices
    Ndata = X.shape[0]
    Ntr = int(np.rint(Ndata*pcSplit))
    Nte = Ndata-Ntr
    np.random.seed(seed)    
    idx = np.random.permutation(Ndata)
    trIdx = idx[:Ntr]
    teIdx = idx[Ntr:]
    # Split data
    xTr = X[trIdx,:]
    yTr = y[trIdx]
    xTe = X[teIdx,:]
    yTe = y[teIdx]
    return xTr,yTr,xTe,yTe,trIdx,teIdx


# Splits data into training and test set, pcSplit defines the percent of
# the data should be used as training data. The major difference to
# trteSplit is that we select the percent from each class individually.
# This means that we are assured to have enough points for each class.
def trteSplitEven(X,y,pcSplit,seed=None):
    labels = np.unique(y)
    xTr = np.zeros((0,X.shape[1]))
    xTe = np.zeros((0,X.shape[1]))
    yTe = np.zeros((0,),dtype=int)
    yTr = np.zeros((0,),dtype=int)
    trIdx = np.zeros((0,),dtype=int)
    teIdx = np.zeros((0,),dtype=int)
    np.random.seed(seed)
    for label in labels:
        classIdx = np.where(y==label)[0]
        NPerClass = len(classIdx)
        Ntr = int(np.rint(NPerClass*pcSplit))
        idx = np.random.permutation(NPerClass)
        trClIdx = classIdx[idx[:Ntr]]
        teClIdx = classIdx[idx[Ntr:]]
        trIdx = np.hstack((trIdx,trClIdx))
        teIdx = np.hstack((teIdx,teClIdx))
        # Split data
        xTr = np.vstack((xTr,X[trClIdx,:]))
        yTr = np.hstack((yTr,y[trClIdx]))
        xTe = np.vstack((xTe,X[teClIdx,:]))
        yTe = np.hstack((yTe,y[teClIdx]))

    return xTr,yTr,xTe,yTe,trIdx,teIdx

def process_training_data(filepath):
    # Read the CSV file
    df = pd.read_csv(filepath)
    
    # Drop any rows with missing values if necessary
    df = df.dropna()
    
    # Convert numeric columns (excluding 'y' and 'x7')
    numeric_cols = df.columns[2:]  # x1 to x13
    numeric_cols = numeric_cols.drop('x7')  # Exclude 'x7' which is categorical
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Encode categorical columns ('y' and 'x7')
    label_enc_y = LabelEncoder()
    label_enc_x7 = LabelEncoder()
    
    df['y'] = label_enc_y.fit_transform(df['y'])  # Encode target variable 'y'
    df['x7'] = label_enc_x7.fit_transform(df['x7'])  # Encode feature 'x7'
    df=df.drop(columns=['x7'])  # Drop 'x7' as it's not needed

    

    return df



def fetchDataset(dataset='iris'):
    if dataset == 'iris':
        X = genfromtxt('irisX.txt', delimiter=',')
        y = genfromtxt('irisY.txt', delimiter=',',dtype=np.int_)-1
        pcadim = 2
    elif dataset == 'wine':
        X = genfromtxt('wineX.txt', delimiter=',')
        y = genfromtxt('wineY.txt', delimiter=',',dtype=np.int_)-1
        pcadim = 0
    elif dataset == 'olivetti':
        X = genfromtxt('olivettifacesX.txt', delimiter=',')
        X = X/255
        y = genfromtxt('olivettifacesY.txt', delimiter=',',dtype=np.int_)
        pcadim = 20
    elif dataset == 'vowel':
        X = genfromtxt('vowelX.txt', delimiter=',')
        y = genfromtxt('vowelY.txt', delimiter=',',dtype=np.int_)
        pcadim = 0
    elif dataset == 'challenge-train':
        # Example usage:
        df = process_training_data('TrainOnMe-c12083a9-22bc-4a0a-949f-77862078ab62.csv')

        labels=['SpaceX', 'Tesla', 'TwitterX']
        # Separate features (X) and target variable (y)
        X = df.drop(columns=['y']).to_numpy()  # Drop 'y' as it's the target
        y = df['y'].to_numpy()  # Target variable
        pcadim=0
    else:
        print("Please specify a dataset!")
        X = np.zeros(0)
        y = np.zeros(0)
        pcadim = 0
    return X,y,pcadim


def genBlobs(n_samples=200,centers=5,n_features=2):
    def genBlobs(n_samples=200, centers=5, n_features=2):
        """
        Generate isotropic Gaussian blobs for clustering.

        Parameters:
        n_samples (int): The total number of points equally divided among clusters. Default is 200.
        centers (int): The number of centers to generate, or the fixed center locations. Default is 5.
        n_features (int): The number of features for each sample. Default is 2.

        Returns:
        tuple: A tuple containing:
            - X (ndarray of shape [n_samples, n_features]): The generated samples.
            - y (ndarray of shape [n_samples]): The integer labels for cluster membership of each sample.
        """
    X, y = make_blobs(n_samples=n_samples, centers=centers, n_features=n_features,random_state=0)
    return X,y


# Scatter plots the two first dimension of the given data matrix X
# and colors the points by the labels.
def scatter2D(X,y):
    labels = np.unique(y)
    Ncolors = len(labels)
    xx = np.arange(Ncolors)
    ys = [i+xx+(i*xx)**2 for i in range(Ncolors)]
    colors = cm.rainbow(np.linspace(0, 1, len(ys)))
    c = 1.0
    for label in labels:
        classIdx = np.where(y==label)[0]
        Xclass = X[classIdx,:]
        plt.scatter(Xclass[:,0],Xclass[:,1],linewidths=1,s=25,color=colors[label],marker='o',alpha=0.75)
        c += 1.

    plt.show()


def plotGaussian(X,y,mu,sigma):
    labels = np.unique(y)
    Ncolors = len(labels)
    xx = np.arange(Ncolors)
    ys = [i+xx+(i*xx)**2 for i in range(Ncolors)]
    colors = cm.rainbow(np.linspace(0, 1, len(ys)))
    c = 1.0
    for label in labels:
        classIdx = y==label
        Xclass = X[classIdx,:]
        plot_cov_ellipse(sigma[label], mu[label])
        plt.scatter(Xclass[:,0],Xclass[:,1],linewidths=1,s=25,color=colors[label],marker='o',alpha=0.75)
        c += 1.

    plt.show()


# The function below, `testClassifier`, will be used to try out the different datasets.
# `fetchDataset` can be provided with any of the dataset arguments `wine`, `iris`, `olivetti` and `vowel`.
# Observe that we split the data into a **training** and a **testing** set.
def testClassifier(classifier, dataset='iris', dim=0, split=0.7, ntrials=100):

    X,y,pcadim = fetchDataset(dataset)

    means = np.zeros(ntrials,);

    for trial in range(ntrials):

        xTr,yTr,xTe,yTe,trIdx,teIdx = trteSplitEven(X,y,split,trial)

        # Do PCA replace default value if user provides it
        if dim > 0:
            pcadim = dim

        if pcadim > 0:
            pca = decomposition.PCA(n_components=pcadim)
            pca.fit(xTr)
            xTr = pca.transform(xTr)
            xTe = pca.transform(xTe)

        # Train
        trained_classifier = classifier.trainClassifier(xTr, yTr)
        # Predict
        yPr = trained_classifier.classify(xTe)

        # Compute classification error
        if trial % 10 == 0:
            print("Trial:",trial,"Accuracy","%.3g" % (100*np.mean((yPr==yTe).astype(float))) )

        means[trial] = 100*np.mean((yPr==yTe).astype(float))

    print("Final mean classification accuracy ", "%.3g" % (np.mean(means)), "with standard deviation", "%.3g" % (np.std(means)))


# ## Plotting the decision boundary
#
# This is some code that you can use for plotting the decision boundary
# boundary in the last part of the lab.
def plotBoundary(classifier, dataset='iris', split=0.7):

    X,y,pcadim = fetchDataset(dataset)
    xTr,yTr,xTe,yTe,trIdx,teIdx = trteSplitEven(X,y,split,1)
    classes = np.unique(y)

    pca = decomposition.PCA(n_components=2)
    pca.fit(xTr)

    xTr = pca.transform(xTr)
    xTe = pca.transform(xTe)

    pX = np.vstack((xTr, xTe))
    py = np.hstack((yTr, yTe))

    # Train
    trained_classifier = classifier.trainClassifier(xTr, yTr)

    xRange = np.arange(np.min(pX[:,0]),np.max(pX[:,0]),np.abs(np.max(pX[:,0])-np.min(pX[:,0]))/100.0)
    yRange = np.arange(np.min(pX[:,1]),np.max(pX[:,1]),np.abs(np.max(pX[:,1])-np.min(pX[:,1]))/100.0)

    grid = np.zeros((yRange.size, xRange.size))

    for (xi, xx) in enumerate(xRange):
        for (yi, yy) in enumerate(yRange):
            # Predict
            grid[yi,xi] = trained_classifier.classify(np.array([[xx, yy]]))

    
    ys = [i+xx+(i*xx)**2 for i in range(len(classes))]
    colormap = cm.rainbow(np.linspace(0, 1, len(ys)))

    fig = plt.figure()
    # plt.hold(True)
    conv = ColorConverter()
    for (color, c) in zip(colormap, classes):
        try:
            CS = plt.contour(xRange,yRange,(grid==c).astype(float),15,linewidths=0.25,colors=conv.to_rgba_array(color))
        except ValueError:
            pass
        trClIdx = np.where(y[trIdx] == c)[0]
        teClIdx = np.where(y[teIdx] == c)[0]
        plt.scatter(xTr[trClIdx,0],xTr[trClIdx,1],marker='o',c=color,s=40,alpha=0.5, label="Class "+str(c)+" Train")
        plt.scatter(xTe[teClIdx,0],xTe[teClIdx,1],marker='*',c=color,s=50,alpha=0.8, label="Class "+str(c)+" Test")
    plt.legend(bbox_to_anchor=(1., 1), loc=2, borderaxespad=0.)
    fig.subplots_adjust(right=0.7)
    plt.title("Decision boundary, dataset "+dataset)
    plt.show()


def visualizeOlivettiVectors(xTr, Xte):
    N = xTr.shape[0]
    Xte = Xte.reshape(64, 64).transpose()
    plt.subplot(1, 2, 1)
    plt.xticks([])
    plt.yticks([])
    plt.title("Test image")
    plt.imshow(Xte, cmap=plt.get_cmap('gray'))
    for i in range(0, N):
        plt.subplot(N, 2, 2+2*i)
        plt.xticks([])
        plt.yticks([])
        plt.title("Matched class training image %i" % (i+1))
        X = xTr[i, :].reshape(64, 64).transpose()
        plt.imshow(X, cmap=plt.get_cmap('gray'))
    plt.show()


class DecisionTreeClassifier(object):
    def __init__(self):
        self.trained = False

    def trainClassifier(self, Xtr, yTr, W=None):
        rtn = DecisionTreeClassifier()
        rtn.classifier = tree.DecisionTreeClassifier(max_depth=int(Xtr.shape[1]/2+1))
        if W is None:
            rtn.classifier.fit(Xtr, yTr)
        else:
            rtn.classifier.fit(Xtr, yTr, sample_weight=W.flatten())
        rtn.trained = True
        return rtn

    def classify(self, X):
        return self.classifier.predict(X)

# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
## FROM LAB 3
def computePrior(labels, W=None):
    Npts = labels.shape[0]
    if W is None:
        W = np.ones((Npts,1))/Npts
    else:
        assert(W.shape[0] == Npts)
    classes = np.unique(labels)
    Nclasses = np.size(classes)

    prior = np.zeros((Nclasses,1))

    # TODO: compute the values of prior for each class!
    # ==========================
    for i, c in enumerate(classes):
        idx = np.where(labels == c)[0]
        W_c = W[idx]
        prior[i] = np.sum(W_c)/np.sum(W)
    # ==========================

    return prior

def cal_Cov(X1, W):
    """
    Calculate the covariance matrix for the given dataset and weights.

    Parameters:
    X1 (numpy.ndarray): A 2D array where each column represents a variable and each row represents an observation.
    W (numpy.ndarray): A 1D array of weights corresponding to each observation.

    Returns:
    numpy.ndarray: A 2D array representing the covariance matrix of the weighted dataset.
    """
    cov = np.zeros((X1.shape[1], X1.shape[1]))
    for i in range(X1.shape[1]):
        centrX = X1[:,i] - np.mean(X1[:,i])
        cov[i, i] = np.dot(np.multiply(centrX, centrX), W)/np.sum(W)
    return cov

def mlParams(X, labels, W=None):
    """
    Compute the mean (mu) and covariance (sigma) for each class in the dataset.
    Parameters:
    X : numpy.ndarray
        A 2D array of shape (Npts, Ndims) containing the data points.
    labels : numpy.ndarray
        A 1D array of shape (Npts,) containing the class labels for each data point.
    W : numpy.ndarray, optional
        A 2D array of shape (Npts, 1) containing the weights for each data point. 
        If None, equal weights are assumed.
    Returns:
    mu : numpy.ndarray
        A 2D array of shape (Nclasses, Ndims) containing the mean vectors for each class.
    sigma : numpy.ndarray
        A 3D array of shape (Nclasses, Ndims, Ndims) containing the covariance matrices for each class.
    """
    assert(X.shape[0]==labels.shape[0])
    Npts,Ndims = np.shape(X)
    classes = np.unique(labels)
    Nclasses = np.size(classes)

    if W is None:
        W = np.ones((Npts,1))/float(Npts)

    mu = np.zeros((Nclasses,Ndims))
    sigma = np.zeros((Nclasses,Ndims,Ndims))

    # TODO: fill in the code to compute mu and sigma!
    # ==========================
    for i, c in enumerate(classes):
        idx = np.where(labels == c)[0]
        X_c = X[idx] 
        W_c = W[idx]
        mu[i, :] = np.dot(np.transpose(W_c), X_c)/np.sum(W_c)
        sigma[i, :, :] = cal_Cov(X_c, W_c)
    
    # ==========================

    return mu, sigma

def classifyBayes(X, prior, mu, sigma):
    """
    Classifies data points using the Bayesian classification rule.
    Parameters:
    X (numpy.ndarray): A 2D array of shape (Npts, Ndims) containing the data points to classify.
    prior (numpy.ndarray): A 1D array of shape (Nclasses,) containing the prior probabilities for each class.
    mu (numpy.ndarray): A 2D array of shape (Nclasses, Ndims) containing the mean vectors for each class.
    sigma (numpy.ndarray): A 3D array of shape (Nclasses, Ndims, Ndims) containing the covariance matrices for each class.
    Returns:
    numpy.ndarray: A 1D array of shape (Npts,) containing the predicted class labels for each data point.
    """

    Npts = X.shape[0]
    Nclasses,Ndims = np.shape(mu)
    logProb = np.zeros((Nclasses, Npts))

    # TODO: fill in the code to compute the log posterior logProb!
    # ==========================
    inv_mat_counter = 0
    for k in range(Nclasses):
        for j in range(Npts):
            xstar = X[j]
            if np.linalg.det(sigma[k]) == 0:
                inv_s = sigma[k]
                inv_mat_counter += 1
            else:                                                                                                              
                inv_s = np.linalg.inv(sigma[k])   
            #inv_s = np.linalg.inv(sigma[k])
            centerx = xstar-mu[k]
            r1 = -0.5*np.log(np.linalg.det(sigma[k])) 
            r2 = -0.5*np.dot(np.dot(centerx, inv_s), np.transpose(centerx))
            r3 = np.log(prior[k])
            logProb[k, j] = r1 + r2 + r3   
    #print("Number of times the covariance matrix was singular: ", inv_mat_counter)   
            
    # ==========================
    
    # one possible way of finding max a-posteriori once
    # you have computed the log posterior
    h = np.argmax(logProb,axis=0)
    return h

# NOTE: no need to touch this
class BayesClassifier(object):
    def __init__(self):
        self.trained = False

    def trainClassifier(self, X, labels, W=None):
        rtn = BayesClassifier()
        rtn.prior = computePrior(labels, W)
        rtn.mu, rtn.sigma = mlParams(X, labels, W)
        rtn.trained = True
        return rtn

    def classify(self, X):
        return classifyBayes(X, self.prior, self.mu, self.sigma)
    
# ------------------------------------------------------------------------------------------------------------------
def trainBoost(base_classifier, X, labels, T=10):
    # these will come in handy later on
    Npts,Ndims = np.shape(X)

    classifiers = [] # append new classifiers to this list
    alphas = [] # append the vote weight of the classifiers to this list

    # The weights for the first iteration
    wCur = np.ones((Npts,1))/float(Npts)

    for i_iter in range(0, T):
        # a new classifier can be trained like this, given the current weights
        classifiers.append(base_classifier.trainClassifier(X, labels, wCur))

        # do classification for each point
        vote = classifiers[-1].classify(X)

        # TODO: Fill in the rest, construct the alphas etc.
        # ==========================
        epsilon = np.dot(wCur.reshape(-1), (1-(vote == labels).astype(int)))
        alpha = 0.5*(np.log(1-epsilon)-np.log(epsilon))
        
        idxT = np.where(vote == labels)
        idxF = np.where(vote != labels)
        
        wCur[idxT] = wCur[idxT] * np.exp(-alpha)
        wCur[idxF] = wCur[idxF] * np.exp(alpha)
        wCur = wCur/np.sum(wCur)

        alphas.append(alpha) # you will need to append the new alpha
        # ==========================
        
    return classifiers, alphas

def classifyBoost(X, classifiers, alphas, Nclasses):
    Npts = X.shape[0]
    Ncomps = len(classifiers)

    # if we only have one classifier, we may just classify directly
    if Ncomps == 1:
        return classifiers[0].classify(X)
    else:
        votes = np.zeros((Npts,Nclasses))

        # TODO: implement classification when we have trained several classifiers!
        # here we can do it by filling in the votes vector with weighted votes
        # ==========================
        for t in range(Ncomps):
            vote = classifiers[t].classify(X)
            for x in range(Npts):
                votes[x, vote[x]] += alphas[t]
        # ==========================

        # one way to compute yPred after accumulating the votes
        return np.argmax(votes,axis=1)
    

# NOTE: no need to touch this
class BoostClassifier(object):
    def __init__(self, base_classifier, T=10):
        self.base_classifier = base_classifier
        self.T = T
        self.trained = False

    def trainClassifier(self, X, labels):
        rtn = BoostClassifier(self.base_classifier, self.T)
        rtn.nbr_classes = np.size(np.unique(labels))
        rtn.classifiers, rtn.alphas = trainBoost(self.base_classifier, X, labels, self.T)
        rtn.trained = True
        return rtn

    def classify(self, X):
        return classifyBoost(X, self.classifiers, self.alphas, self.nbr_classes)