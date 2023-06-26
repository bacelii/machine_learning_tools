'''



Purpose: Storing models that were
implemented in sklearn and tested/created easier api




Notes: 
model.predict --> predicts results
model.coef_ --> coefficients
model.interecpt_
model.score(X,y) --> gives the r2 of the prediction
model.alpha_ --> the LaGrange multiplier after the fit




'''
from sklearn import ensemble 
from sklearn import linear_model
from sklearn import svm
from sklearn import tree 
from sklearn.inspection import permutation_importance
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import numpy as np
import time

def clf_name(clf):
    return str(clf).split("(")[0]

def compute_class_weight(y):
    class_weights = class_weight.compute_class_weight('balanced',
                                                 classes=np.unique(y),
                                                 y=y)
    return class_weights
# ------- Linear Regression ------------
"""
Notes: 
alpha in most models is the LaGrange lambda

"""

def LinearRegression(**kwargs):
    return linear_model.LinearRegression(**kwargs)

def residuals(model,x,y,return_score = True,):
    model.fit(x,y)
    y_pred = model.predict(x)
    residuals = y_pred - y
    
    if return_score:
        score = model.score(x,y)
        return residuals,score
    else:
        return residuals

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


#================= ============== Tree based models ================= ==============

ensemble.RandomForestClassifier
ensemble.BaggingClassifier
ensemble.AdaBoostClassifier
tree.DecisionTreeClassifier


def DecisionTreeClassifier(
    splitter="best",
    criterion = "gini",
    max_depth= None,
    max_features = None,
    random_state=None,
    class_weight = None,
    **kwargs
    ):
    
    """
    
    
    DecisiionTreeClassifier parameters:

    ------- DecisionTree specific parameters ----------
    splitter: default="best" how the split will be chosen
    - "best": best split
    - "random": best random split

    ------- generic tree parameters ---------
    criterion : how the best split is determined
    - 'gini'
    - 'entropy'
    max_depth: int , max depth of trees
    max_features: number of features to look at when doing best split
    - 'auto':sqrt of number of n_features
    - 'sqrt'
    - log2
    - int: specifying the exact numbe of features
    - float: secifying the percentage of total n_features
    - None: always has maximum number of features
    class_weight: dict,list of dict, "balanced"  will add weighting to class decision
    - "balanced" will automatically balance based on composition of y


    """
    
    return tree.DecisionTreeClassifier(
    splitter=splitter,
    criterion = criterion,
    max_depth= max_depth,
    max_features = max_features,
    random_state=random_state,
    class_weight = class_weight,
    **kwargs
    )

tree_parameters = ["max_depth"]
def BaggingClassifier(
    base_estimator = None,
    n_estimators= 10,
    max_samples = 1.0,
    max_features = 1.0,
    bootstrap = True,
    bootstrap_features = False,
    oob_score = False,
    random_state = True,
    verbose = True,
    
    max_depth= None,
    **kwargs):

    """
    Purpose: 
    1) Fits classifiers based on random subsets of the original data
    (done using boostrapping method that uses sampling with replacement)
    2) Aggregates the individual predictions of classifiers 9voting or average)

    Application: to reduce varaince of an estimator

    Baggin Classifier parameters

    --- Bagging specific parameters -------
    base_estimator: Object, default = DecisionTreeClassifier
        the model used that will be trained many times
    n_estimators: int , 
        Number of estimators ensemble will use
    max_samples: int/float, default = 1.0
    - int: only will draw that many samples
    - float: that proportion of saples will draw
    max_features: int/float, default = 1.0
        number of features drawn when creating the training set for that iteration

    boostrap: bool, default=True
        whether samples are drawn with replacement or not
    boostrap_features: bool, default = False
        to draw features with replacement
    oob_score: bool, default=False
        to use a out o bag samples to calculate a generalization error as train
    random_state: int, default = None
    verbose: int, default = None

    """
    tree_dict = dict()
    for k,v in kwargs.items():
        if k in tree_parameters:
            tree_dict[k] = v
    
    if len(tree_dict) > 0 and base_estimator is None:
        base_estimator = sklm.DecisionTreeClassifier(**tree_dict)
    
    
    return ensemble.BaggingClassifier(
    base_estimator = base_estimator,
    n_estimators= n_estimators,
    max_samples = max_samples,
    max_features = max_features,
    bootstrap = bootstrap,
    bootstrap_features = bootstrap_features,
    oob_score = oob_score,
    random_state = random_state,
    verbose = verbose,
    **kwargs
    )



def RandomForestClassifier(
    n_estimators=30,
    criterion = "gini",
    max_depth= None,
    max_features = "auto",
    bootstrap = True,
    oob_score = True,
    random_state=None,
    verbose = False,
    max_samples = None,
    class_weight = None,
    **kwargs
    ):
    
    """
    Purpose: Where the number of features trained on is a subset of overall
    and the samples trained on are a boostrapped samples
    
    ---------RandomForrest specific parameters---------: 
    n_estimators: number o trees to use in the forest
    bootstrap: bool(True): whether bootstrap sapling are used to build trees
    (if not then the whole dataset is used)

    oob_score: bool (False): whether to use out-of =-bag samples to estimate the generalization
    score
    
    max_samples: Number of samples to draw if doing bootstrapping
    verbose
    
    
    ------- generic tree parameters ---------
    criterion : how the best split is determined
    - 'gini'
    - 'entropy'
    max_depth: int , max depth of trees
    max_features: number of features to look at when doing best split
    - 'auto':sqrt of number of n_features
    - 'sqrt'
    - log2
    - int: specifying the exact numbe of features
    - float: secifying the percentage of total n_features
    - None: always has maximum number of features
    class_weight: dict,list of dict, "balanced"  will add weighting to class decision
    - "balanced" will automatically balance based on composition of y
    
    
    
    Example: 
    clf = sklm.RandomForestClassifier(max_depth=5)
    clf.fit(X_train,y_train)
    print(sklu.accuracy(clf,X_test,y_test),clf.oob_score_)
    _ = sklm.feature_importance(clf_for,return_std=True,plot=True)
    """
    
    if max_samples is not None and max_samples >= 1:
        max_samples = None
    
    return ensemble.RandomForestClassifier(
        n_estimators=n_estimators,
        criterion = criterion,
        max_depth= max_depth,
        max_features = max_features,
        bootstrap = bootstrap,
        oob_score = oob_score,
        random_state=random_state,
        verbose = verbose,
        max_samples = max_samples,
        **kwargs
    )


def AdaBoostClassifier(
    base_estimator=None,
    n_estimators = 50,
    learning_rate = 1.0,
    random_state = None,
    **kwargs):

    """
    Purpose: To perform boosting sequential ensembles
    where the subsequent models are trained on weighted 
    data where data missed in previous mehtod are more highly weighted

    ---- AdaBoost specific parameters -----
    base_estimator: Object, default=None
        the estimator used for the ensemble, 
        if None then it is Tree with max_depth = 1
    n_estimators: int, default = 50
        maximum number of estimators used before termination, but
        learning could be terminated earlier (just max possible)
    learning_rate: float, default = 1.0
        weight applied to each classifier at each iteration, 
        so the lower the learning rate the more models would have
    random_state: int
        controls seed given to each base_estimator at each boosting iteration
        (AKA the base estimator has to have a random_state arpument)


    attributes: 
    base_estimator_:Object
        base from which estimators were built
    estimators_: list of objects
        list of the fitted sub estimators
    estimator_weights_: array of floats
        the weights given for each estimator
    estimator_errors_: array of floats
        classification error for each estimator in the  
    feature_importances_: array of floats:
        impurity-based feautre importances
    n_features_in_:
        number of features seend during fit
    feature_names_in:
        names of the features seen during the fit

    """
    tree_dict = dict()
    for k,v in kwargs.items():
        if k in tree_parameters:
            tree_dict[k] = kwargs[k]
            
    for k in tree_dict.keys():
        del kwargs[k]

    
    if len(tree_dict) > 0 and base_estimator is None:
        base_estimator = sklm.DecisionTreeClassifier(**tree_dict)
        
    return ensemble.AdaBoostClassifier(
        base_estimator=base_estimator,
        n_estimators = n_estimators,
        learning_rate = learning_rate,
        random_state = random_state,
        **kwargs)


def GradientBoostingClassifier(
    loss="deviance",
    learning_rate = 0.1,
    n_estimators=100,
    subsample=1.0,
    max_depth = 3,
    random_state=None,
    max_features=None,
    verbose = 0,
    ):
    """
    GradientBoosting

    Purpose: to optimize a certain loss function. Idea is to 
    fit classifiers that go downhill in the gradient but not fit all the way.
    This makes the earning slower and harder to overfit

    Procedure:
    For many stages
    1) regression tres are fit on the negative gradient of the loss
    2) only wegihts the classifiers by a learning rate so not learn to quickly
    3) continue to the next stage and learn on the subsequent gradients

    Application: Usually pretty good for overfitting

    --- GradientBoosting specific parameters ---
    loss: str, default = "deviance"
        the loss function that should be optimized
        - "deviance": logistic regression loss (with probabilistic outputs)
        - "exponential": this is ust the adaboost algorithm
    learning_rate: float default = 0.1,
        how much each of the estimators contributes
    n_estimators: int, default = 100
    subsample:float, default = 1.0
        if less than 1 then will do stochastic gradient boosting where not look
        at all of the samples


    ------- generic tree parameters ---------
    max_depth: int , max depth of trees
    max_features: number of features to look at when doing best split
    - 'auto':sqrt of number of n_features
    - 'sqrt'
    - log2
    - int: specifying the exact numbe of features
    - float: secifying the percentage of total n_features
    - None: always has maximum number of features


    What it returns: 
    n_stimators: int
        number estimators made 
    feature_importances: array
        impurity based feature importances
    oob_importovement: 
        the improvement in loss on the out of bag sample relative to previous iteration
        (ONLY AVAILABLE IF SUBSAMPLE < 1.0)
    train_score: array
        ith train score is the deviance of model iteration i on the in-bag sample 
        (if subsample == 1, then this is the deviance on the training data)
    estimators: array of DecisionTreeregression

    """

    return ensemble.GradientBoostingClassifier(
    loss=loss,
    learning_rate = learning_rate,
    n_estimators=n_estimators,
    subsample=subsample,
    max_depth = max_depth,
    random_state=random_state,
    max_features=max_features,
    verbose = verbose,
    )

def oob_score(clf):
    """
    Purpose: Returns the out of bag error
    for ensembles that use bootstrapping method
    """
    return clf.oob_score_


def is_ensemble(clf):
    if "ensemble" in str(type(clf)):
        return True
    else:
        return False

def feature_importances(
    clf,
    verbose = True,
    plot = False,
    feature_names = None,
    return_std = False,
    method="impurity_decrease",
    #arguments for permutation
    X_permuation=None,
    y_permutation=None,
    n_repeats=10,
    random_state= None,
    
    **kwargs
    ):
    """
    Purpose: Will return the feature importance
    of a tree based classifier
    
    sklm.feature_importances(clf,
                        #method=,
                        verbose = True,
                        plot=True,
                        X_permuation=X_test,
                        y_permutation=y_test,
                        n_repeats=1)
    
    """

    st = time.time()
    
    if verbose:
        print(f"Using method: {method}")
    
    if method =="impurity_decrease":
        try:
            importances = clf.feature_importances_
        except:
            importances = np.mean([
                tree.feature_importances_ for tree in clf.estimators_
            ], axis=0)
            
        if not sklm.is_ensemble(clf):
            std = np.zeros(clf.n_features_in_)
        else:
            try:
                std = np.std([tree.feature_importances_ for tree in clf.estimators_],axis=0)
            except:
                std = np.std([
                    tree.feature_importances_ for tree in clf.estimators_
                ], axis=0)
                
    elif method=="permutation":
        if X_permuation is None or y_permutation is None:
            raise Exception("Trying to do permutation feature importance but no training data")
        result = permutation_importance(
            clf,
            X_permuation,
            y_permutation,
            n_repeats=n_repeats,
            random_state= random_state,
        )
        importances = result.importances_mean
        std = result.importances_std
    else:
        raise Exception(f"Unimplemented importance method: {method}")
        
    elapsed_time = time.time() - st
    if verbose:
        print(f"Time for importances = {elapsed_time}")
        
    if plot:
        vml.plot_feature_importance(clf,feature_names,
                                    title = f"Feature Importance \n({method})",
                                    importances=importances,
                                    std=std,
                                    **kwargs)
    
    if return_std:
        return importances,std
    else:
        return importances
    


#--- from machine_learning_tools ---
from . import pandas_ml as pdml
from . import visualizations_ml as vml

from . import sklearn_models as sklm