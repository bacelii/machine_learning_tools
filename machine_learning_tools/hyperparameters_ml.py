
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import time

def best_hyperparams_RandomizedSearchCV(
    clf,
    parameter_dict,
    X,
    y,
    n_iter_search = 2,
    return_clf = False,
    return_cv_results = True,
    verbose = True,
    n_cv_folds = 5,
    n_jobs = 1):
    """
    Purpose: To find the best parameters in from a 
    random search of a parameter map defined by a dict

    Source: https://scikit-learn.org/stable/auto_examples/model_selection/plot_randomized_search.html#sphx-glr-auto-examples-model-selection-plot-randomized-search-py


    """
    def report(results, n_top=3):
        for i in range(1, n_top + 1):
            candidates = np.flatnonzero(results["rank_test_score"] == i)
            for candidate in candidates:
                print("Model with rank: {0}".format(i))
                print(
                    "Mean validation score: {0:.3f} (std: {1:.3f})".format(
                        results["mean_test_score"][candidate],
                        results["std_test_score"][candidate],
                    )
                )
                print("Parameters: {0}".format(results["params"][candidate]))
                print("")


    random_search = RandomizedSearchCV(
        clf, 
        param_distributions=parameter_dict, #parameter dict to iterate over
        n_iter=n_iter_search,
        verbose = verbose,
        n_jobs=n_jobs,
        cv = n_cv_folds,
    )

    start = time.time()
    random_search.fit(X, y)
    
    if verbose:
        print(
            "RandomizedSearchCV took %.2f seconds for %d candidates parameter settings."
            % ((time.time() - start), n_iter_search)
        )

        report(random_search.cv_results_)
        
    if return_clf:
        if return_cv_results:
            return random_search.best_estimator_,random_search
        else:
            return random_search.best_params_
    else:
        if return_cv_results:
            return random_search.best_params_,random_search
        else:
            return random_search.best_params_


