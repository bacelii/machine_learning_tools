"""
Purpose: Storing models that were
implemented in sklearn and tested/created easier api




Notes: 
model.predict --> predicts results
model.coef_ --> coefficients
model.interecpt_
model.score(X,y) --> gives the r2 of the prediction
model.alpha_ --> the LaGrange multiplier after the fit

"""
import pandas_ml as pdml

def clf_name(clf):
    return str(clf).split("(")[0]


# ------- Linear Regression ------------
"""
Notes: 
alpha in most models is the LaGrange lambda

"""
from sklearn import linear_model

def LinearRegression(**kwargs):
    return linear_model.LinearRegression(**kwargs)

fit_intercept_default = False
alpha_default = 1
l1_ratio_default = 0.5
default_cv_n_splits = 10
    
def ElasticNetCV(l1_ratio = l1_ratio_default,
                fit_intercept = fit_intercept_default,
                 cv_n_splits = default_cv_n_splits,
                **kwargs):
    """
    Purpose: Model that has a mix of L1 and L2 regularization
    and chooses the lamda (called alpha) based on cross 
    validation later when it is fitted
    
    """
    return linear_model.ElasticNetCV(l1_ratio=l1_ratio,
                   fit_intercept=fit_intercept,
                                     cv = cv_n_splits,
                                    **kwargs)

def ElasticNet(alpha=alpha_default,
               l1_ratio = l1_ratio_default,
                fit_intercept = fit_intercept_default,
                **kwargs):
    """
    Purpose: Model that has a mix of L1 and L2 regularization
    and chooses the lamda (called alpha) based on cross 
    validation later when it is fitted
    
    """
    return linear_model.ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        fit_intercept=fit_intercept,
        **kwargs)

def LassoCV(fit_intercept = fit_intercept_default,
            cv_n_splits = default_cv_n_splits,
           **kwargs):
    return linear_model.LassoCV(
        fit_intercept = fit_intercept,
        cv = cv_n_splits,
    )

def Lasso(
    alpha=alpha_default,
    fit_intercept = fit_intercept_default,
    **kwargs):
    
    return linear_model.Lasso(
        alpha=alpha,
        fit_intercept = fit_intercept
    )
    
def RidgeCV(
    fit_intercept = fit_intercept_default,
    cv_n_splits = default_cv_n_splits,
    **kwargs):
    return linear_model.RidgeCV(
        fit_intercept = fit_intercept,
        cv = cv_n_splits
    )
    
def Ridge(
    alpha = alpha_default,
    fit_intercept = fit_intercept_default,
    **kwargs):
    return linear_model.Ridge(
        alpha = alpha,
        fit_intercept = fit_intercept
    )

import numpy as np


def AdaptiveLasso(X,y,
                  CV = True,
                  cv_n_splits = default_cv_n_splits,
                  fit_intercept = fit_intercept_default,
                  alpha = None,
                 coef = None,#the real coefficients if know those
                  verbose = False,
                  n_lasso_iterations = 5,
                 ):
    """
    Example of adaptive Lasso to produce event sparser solutions

    Adaptive lasso consists in computing many Lasso with feature
    reweighting. It's also known as iterated L1.
    
    Help with the implementation: 

    https://gist.github.com/agramfort/1610922

    
    --- Example 1: Using generated data -----
    
    from sklearn.datasets import make_regression
    X, y, coef = make_regression(n_samples=306, n_features=8000, n_informative=50,
                    noise=0.1, shuffle=True, coef=True, random_state=42)

    X /= np.sum(X ** 2, axis=0)  # scale features
    alpha = 0.1
    
    model_al = sklm.AdaptiveLasso(
        X,
        y,
        alpha = alpha,
        coef = coef,
        verbose = True
    )
    
    ---- Example 2: Using simpler data ----
    X,y = pdml.X_y(df_scaled,target_name)
    model_al = sklm.AdaptiveLasso(
        X,
        y,
        verbose = True
    )
    

    """
    if "pandas" in str(type(X)):
        X = X.to_numpy()
    if "pandas" in str(type(y)):
        y = y.to_numpy()
    

    # function that computes the absolute value square root of an input
    def g(w):
        return np.sqrt(np.abs(w))
    
    #computes 1/(2*square_root(abs(w)))
    def gprime(w):
        return 1. / (2. * np.sqrt(np.abs(w)) + np.finfo(float).eps)

    # Or another option:
    # ll = 0.01
    # g = lambda w: np.log(ll + np.abs(w))
    # gprime = lambda w: 1. / (ll + np.abs(w))

    n_samples, n_features = X.shape
    def p_obj(w,alpha):
        return 1. / (2 * n_samples) * np.sum((y - np.dot(X, w)) ** 2) + alpha * np.sum(g(w))

    weights = np.ones(n_features)
    

    for k in range(n_lasso_iterations):
        X_w = X / weights[np.newaxis, :]
        if CV:
            clf = linear_model.LassoCV(
                #alpha=alpha, 
                fit_intercept=fit_intercept,
                cv = cv_n_splits)
            
        else:
            clf = linear_model.Lasso(
                    alpha = alpha,
                    fit_intercept = fit_intercept
            )
        clf.fit(X_w, y)
        if CV:
            curr_alpha = clf.alpha_
        else:
            curr_alpha = alpha
        
        coef_ = clf.coef_ / weights
        weights = gprime(coef_)
        if verbose:
            print(p_obj(coef_,curr_alpha))  # should go down

    clf.coef_ = coef_
    if verbose:
        X_w = X / weights[np.newaxis, :]
        
        print(f"Final R^2 score: {clf.score(X,y)}")
    #print(np.mean((clf.coef_ != 0.0) == (coef != 0.0)))
    
    return clf

import numpy as np
def ranked_features(model,
                    feature_names=None,
                   verbose = False):
    """
    Purpose: to return the features (or feature)
    indexes that are most important by the absolute values
    of their coefficients
    """
    feature_coef = np.abs(model.coef_)
    if verbose:
        if feature_names is not None:
            print(f"feature_names = {feature_names}")
        print(f"feature_weights= {feature_coef}")
        
    
    ordered_idx = np.flip(np.argsort(feature_coef))
#     if verbose:
#         print(f"ordered_idx= {ordered_idx}")
        
    if feature_names is not None:
        ordered_features = np.array(feature_names)[ordered_idx]
        if verbose:
            print(f"\nordered_features = {ordered_features}")
        return ordered_features
            
    else:
        return ordered_idx
     
    

    
# ---------- Modeul visualizations for those with lambda parametere ----
import numpy as np
import matplotlib.pyplot as plt

def set_legend_outside_plot(ax,scale_down=0.8):
    """
    Will adjust your axis so that the legend appears outside of the box
    """
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * scale_down, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    return ax

def plot_regularization_paths(
    model_func,
    df = None,
    target_name = None,
    X =None,
    y = None,
    n_alphas = 200,
    alph_log_min = -1,
    alpha_log_max = 5,
    reverse_axis = True,
    model_func_kwargs = None,
    ):
    """
    Purpose: Will plot the regularization paths
    for a certain model
    
    # Author: Fabian Pedregosa -- <fabian.pedregosa@inria.fr>
    # License: BSD 3 clause
    
    
    Example from oneline: 
    
    # X is the 10x10 Hilbert matrix
    X = 1. / (np.arange(1, 11) + np.arange(0, 10)[:, np.newaxis])
    y = np.ones(10)
    
    plot_regularization_paths(
    sklm.Ridge,
    X = X,
    y =y,
    alph_log_min = -10,
    alpha_log_max = -2,
    )
    """
    if model_func_kwargs is None:
        model_func_kwargs = dict()
    
    if X is None or y is None:
        X,y = pdml.X_y(df,target_name)

    # ################################
    # Compute paths


    alphas = np.logspace(alph_log_min, alpha_log_max, n_alphas)
    coefs = []
    for a in alphas:
        ridge = model_func(alpha=a, fit_intercept=False,**model_func_kwargs)
        ridge.fit(X, y)
        coefs.append(ridge.coef_)

        
    coefs = np.array(coefs)
    # ##################################
    # Display results

    ax = plt.gca()

    
    for j,feat_name in enumerate(pdml.feature_names(X)):
        ax.plot(alphas,coefs[:,j],label = feat_name)
    ax.set_xscale('log')
    if reverse_axis:
        ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
    plt.xlabel('alpha')
    plt.ylabel('weights')
    plt.title('Ridge coefficients as a function of the regularization')
    plt.axis('tight')
    plt.legend()
    
    set_legend_outside_plot(ax)
    plt.show()
    
    
def coef_summary(feature_names,
                 model = None,
                 coef_ = None,
                 intercept = True):
    if coef_ is None:
        coef_ = model.coef_
    if intercept:
        print(f"Intercept: {model.intercept_}")
    for f,c in zip(feature_names,coef_):
        print(f"{f}:{c}")
        
        
# =========================== CLASSIFIERS ================
def classes(clf):
    return clf.classes_

def n_features_in_(clf):
    return clf.n_features_in_

def LogisticRegression(**kwargs):
    """
    This one you can set the coefficients of the linear classifier
    """
    return linear_model.LogisticRegression(**kwargs)

from sklearn import svm

def SVC(kernel="rbf",
        C = 1,
        gamma = "scale",
        **kwargs):
    """
    SVM with the possibilities of adding kernels
    
    """
    return svm.SVC(kernel=kernel,
                   C = C,
                   gamma = gamma,
                   **kwargs)
