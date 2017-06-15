'''
Created on Jun 11, 2017

@author: BihuiJin
'''
import numpy as np
from sklearn.utils import check_X_y
#from sklearn.externals.joblib import Parallel, delayed
from sklearn.base import clone
#from ..base import is_classifier
#from ..model_selection import check_cv
from sklearn.model_selection._validation import _score
from sklearn.metrics.scorer import check_scoring
from sklearn.model_selection import GridSearchCV

def _forward_search_single_fit(forward_search, estimator, X, y, scorer):
    """
    Return the score for a fit across one fold.
    """
    return forward_search._fit(
        X, y, lambda estimator, features:
        _score(estimator, X[:, features], y, scorer)).scores_

class ForwardSearch(object):
    '''
    Parameters
    ----------
    estimator : object
        A supervised learning estimator with a `fit` method that updates a
        `coef_` attribute that holds the fitted parameters. Important features
        must correspond to high absolute values in the `coef_` array.

        For instance, this is the case for most supervised learning
        algorithms such as Support Vector Classifiers and Generalized
        Linear Models from the `svm` and `linear_model` modules.

    param_grid : dict or list of dictionaries
        Dictionary with parameters names (string) as keys and lists of parameter 
        settings to try as values, or a list of such dictionaries, in which case 
        the grids spanned by each dictionary in the list are explored. This enables 
        searching over any sequence of parameter settings.

    n_features_to_select : int or None (default=None)
        The number of features to select. If `None`, half of the features
        are selected.

    step : int or float, optional (default=1)
        If greater than or equal to 1, then `step` corresponds to the (integer)
        number of features to remove at each iteration.
        If within (0.0, 1.0), then `step` corresponds to the percentage
        (rounded down) of features to remove at each iteration.

    verbose : int, default=0
        Controls verbosity of output.
    '''
    
    def __init__(self, estimator, param_grid, n_features_to_select=None, step=1,
             verbose=0):
        self.estimator = estimator
        self.param_grid = param_grid
        self.n_features_to_select = n_features_to_select
        self.step = step
        self.verbose = verbose
            
    def fit(self, X, y, n_features_to_select=None):
        """Fit the forward search model and then the underlying estimator on the selected
           features.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values.
        """
        if n_features_to_select is not None:
            self.n_features_to_select = n_features_to_select
        return self._fit(X, y)
        
    def _fit(self, X, y, step_score=None):
        X, y = check_X_y(X, y, "csc")
        # Initialization
        n_features = X.shape[1]
        if self.n_features_to_select is None:
            n_features_to_select = n_features // 2
        else:
            n_features_to_select = self.n_features_to_select

        if 0.0 < self.step < 1.0:
            step = int(max(1, self.step * n_features))
        else:
            step = int(self.step)
        if step <= 0:
            raise ValueError("Step must be >0")

        # true means haven't added into the pool of selected features.
        support_ = np.zeros(n_features, dtype=np.bool)
        ranking_ = np.ones(n_features, dtype=np.int)

        self.scores_ = []

        # Elimination
        while n_features_to_select > np.sum(support_):
            # Remaining features
            # np.logical_not(support_) means features which not haven't been selected.
            features = np.arange(n_features)[np.logical_not(support_)]
            #print('features 1 %s'%features)
            features_scores = np.zeros(n_features, dtype=np.float)[np.logical_not(support_)]
            
            for index, feature in enumerate(features):
                features_support = np.array(support_)
                features_support[feature] = True
                # Rank the remaining features
                estimator = clone(GridSearchCV(self.estimator, self.param_grid))
                if self.verbose > 0:
                    print("Fitting estimator with %d features." % np.sum(support_))
                
                estimator.fit(X[:, features_support], y)
                features_scores[index] = estimator.best_score_
            # Get ranks
            if features_scores.ndim > 1:
                ranks = np.argsort(features_scores.sum(axis=0))
            else:
                # set index from high to low because of '-'. np.argsort will sort array from low to high.
                ranks = np.argsort(-features_scores)

            # for sparse case ranks is matrix
            ranks = np.ravel(ranks)
            
            # Eliminate the best features
            threshold = min(step, n_features_to_select - np.sum(support_))

            support_[features[ranks][:threshold]] = True
            # lower means more import
            ranking_[np.logical_not(support_)] += 1
            
            # Compute step score on the selection iteration
            self.scores_.append(features_scores[ranks][0])

        # Set final attributes
        features = np.arange(n_features)[support_]
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X[:, features], y)

        self.n_features_ = support_.sum()
        self.support_ = support_
        self.ranking_ = ranking_

        return self

class FeatureSearch(object):
    '''
    Parameters
    ----------
    estimator : object
        A supervised learning estimator with a `fit` method that updates a
        `coef_` attribute that holds the fitted parameters. Important features
        must correspond to high absolute values in the `coef_` array.

        For instance, this is the case for most supervised learning
        algorithms such as Support Vector Classifiers and Generalized
        Linear Models from the `svm` and `linear_model` modules.
        
    param_grid : dict or list of dictionaries
        Dictionary with parameters names (string) as keys and lists of parameter 
        settings to try as values, or a list of such dictionaries, in which case 
        the grids spanned by each dictionary in the list are explored. This enables 
        searching over any sequence of parameter settings.

    step : int or float, optional (default=1)
        If greater than or equal to 1, then `step` corresponds to the (integer)
        number of features to remove at each iteration.
        If within (0.0, 1.0), then `step` corresponds to the percentage
        (rounded down) of features to remove at each iteration.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 3-fold cross-validation,
        - integer, to specify the number of folds.
        - An object to be used as a cross-validation generator.
        - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`sklearn.model_selection.StratifiedKFold` is used. If the 
        estimator is a classifier or if ``y`` is neither binary nor multiclass, 
        :class:`sklearn.model_selection.KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

    scoring : string, callable or None, optional, default: None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.

    verbose : int, default=0
        Controls verbosity of output.

    n_jobs : int, default 1
        Number of cores to run in parallel while fitting across folds.
        Defaults to 1 core. If `n_jobs=-1`, then number of jobs is set
        to number of cores.
        
    search_strategy : 'forward' or 'backward'

    '''
    def __init__(self, estimator, param_grid, step=1, cv=None, scoring=None, n_jobs=1, search_strategy='forward'):
        self.estimator = estimator
        self.param_grid = param_grid
        self.step = step
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.search_strategy = search_strategy

    def fit(self, X, y):
        X, y = check_X_y(X, y, "csr")
        
        # Initialization
        scorer = check_scoring(self.estimator, scoring=self.scoring)
        n_features = X.shape[1]
        n_features_to_select = n_features
        if 0.0 < self.step < 1.0:
            step = int(max(1, self.step * n_features))
        else:
            step = int(self.step)
        if step <= 0:
            raise ValueError("Step must be >0")
        
        forward_search = ForwardSearch(self.estimator, self.param_grid, n_features_to_select, step=self.step)
        scores = _forward_search_single_fit(forward_search, self.estimator, X, y, scorer)
        # Determine the number of subsets of features by fitting across
        # the train folds and choosing the "features_to_select" parameter
        # that gives the least averaged error across all folds.

        # Note that joblib raises a non-picklable error for bound methods
        # even if n_jobs is set to 1 with the default multiprocessing
        # backend.
        # This branching is done so that to
        # make sure that user code that sets n_jobs to 1
        # and provides bound methods as scorers is not broken with the
        # addition of n_jobs parameter in version 0.18.
        '''
        if self.search_strategy == 'forward':
            self.single_fit = _forward_search_single_fit
        if self.n_jobs == 1:
            parallel, func = list, _forward_search_single_fit
        else:
            parallel, func, = Parallel(n_jobs=self.n_jobs), delayed(self.single_fit)

        scores = parallel(
            func(forward_search, self.estimator, X, y, train, test, scorer)
            for train, test in cv.split(X, y))
        '''
        
        n_features_to_select = max(
            n_features - (np.argmax(scores) * step),
            n_features_to_select)
        # Re-execute an elimination with best_k over the whole set
        forward_search_final = ForwardSearch(self.estimator, self.param_grid, n_features_to_select, step=self.step)

        forward_search_final.fit(X, y)

        # Set final attributes
        self.support_ = forward_search_final.support_
        self.n_features_ = forward_search_final.n_features_
        self.ranking_ = forward_search_final.ranking_
        
        return self
        
if __name__ == "__main__":
    from sklearn import svm, datasets
    iris = datasets.load_iris()
    parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
    svr = svm.SVC()
    #forward_search = FeatureSearch(svr, parameters)
    #forward_search = ForwardSearch(svr, parameters)
    #forward_search.fit(iris.data, iris.target)
    feature_search = FeatureSearch(svr, parameters, step=1,
                  scoring='accuracy')
    feature_search.fit(iris.data, iris.target)
    print(feature_search.n_features_)
        