'''
Created on Jun 11, 2017

@author: BihuiJin
'''
import numpy as np
from sklearn.utils import check_X_y
from sklearn.externals.joblib import Parallel, delayed
#from ..base import is_classifier
#from ..model_selection import check_cv
#from ..metrics.scorer import check_scoring
from sklearn.model_selection import GridSearchCV

class FowardSearch(object):
    
    def __init__(self, estimator, param_grid, n_features_to_select=None, step=1,
             verbose=0):
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.step = step
        self.verbose = verbose
        self.grid_search = GridSearchCV(estimator, param_grid)
            
        def fit(self, X, y):
            """Fit the forward search model and then the underlying estimator on the selected
               features.
    
            Parameters
            ----------
            X : {array-like, sparse matrix}, shape = [n_samples, n_features]
                The training input samples.
    
            y : array-like, shape = [n_samples]
                The target values.
            """
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

        support_ = np.ones(n_features, dtype=np.bool)
        ranking_ = np.ones(n_features, dtype=np.int)

        if step_score:
            self.scores_ = []

        # Elimination
        while np.sum(support_) > n_features_to_select:
            # Remaining features
            features = np.arange(n_features)[support_]

            # Rank the remaining features
            estimator = clone(self.estimator)
            if self.verbose > 0:
                print("Fitting estimator with %d features." % np.sum(support_))

            estimator.fit(X[:, features], y)

            # Get coefs
            if hasattr(estimator, 'coef_'):
                coefs = estimator.coef_
            else:
                coefs = getattr(estimator, 'feature_importances_', None)
            if coefs is None:
                raise RuntimeError('The classifier does not expose '
                                   '"coef_" or "feature_importances_" '
                                   'attributes')

            # Get ranks
            if coefs.ndim > 1:
                ranks = np.argsort(safe_sqr(coefs).sum(axis=0))
            else:
                ranks = np.argsort(safe_sqr(coefs))

            # for sparse case ranks is matrix
            ranks = np.ravel(ranks)

            # Eliminate the worse features
            threshold = min(step, np.sum(support_) - n_features_to_select)

            # Compute step score on the previous selection iteration
            # because 'estimator' must use features
            # that have not been eliminated yet
            if step_score:
                self.scores_.append(step_score(estimator, features))
            support_[features[ranks][:threshold]] = False
            ranking_[np.logical_not(support_)] += 1

        # Set final attributes
        features = np.arange(n_features)[support_]
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X[:, features], y)

        # Compute step score when only n_features_to_select features left
        if step_score:
            self.scores_.append(step_score(self.estimator_, features))
        self.n_features_ = support_.sum()
        self.support_ = support_
        self.ranking_ = ranking_

        return self

class FeatureSearch(object):

    def __init__(self, estimator, param_grid, step=1, cv=None, scoring=None, n_jobs=1):
        self.estimator = estimator
        self.param_grid = param_grid
        self.step = step
        self.cv = cv
        self.scoring = scoring
        self.grid_search = GridSearchCV(estimator, param_grid)
        self.n_jobs = n_jobs

    def fit(self, X, y):
        X, y = check_X_y(X, y, "csr")
        self.grid_search_result = self.grid_search.fit(X, y)
        print(self.grid_search_result)
        print(self.grid_search_result.best_params_)
        print(self.grid_search_result.best_score_)
        print(X)
        print(y)
        
        # Initialization
        n_features = X.shape[1]
        n_features_to_select = 1
        if 0.0 < self.step < 1.0:
            step = int(max(1, self.step * n_features))
        else:
            step = int(self.step)
        if step <= 0:
            raise ValueError("Step must be >0")
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

        if self.n_jobs == 1:
            parallel, func = list, _rfe_single_fit
        else:
            parallel, func, = Parallel(n_jobs=self.n_jobs), delayed(_rfe_single_fit)

        scores = parallel(
            func(rfe, self.estimator, X, y, train, test, scorer)
            for train, test in cv.split(X, y))

        scores = np.sum(scores, axis=0)
        n_features_to_select = max(
            n_features - (np.argmax(scores) * step),
            n_features_to_select)

        # Re-execute an elimination with best_k over the whole set
        rfe = RFE(estimator=self.estimator,
                  n_features_to_select=n_features_to_select, step=self.step)

        rfe.fit(X, y)

        # Set final attributes
        self.support_ = rfe.support_
        self.n_features_ = rfe.n_features_
        self.ranking_ = rfe.ranking_
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(self.transform(X), y)

        # Fixing a normalization error, n is equal to get_n_splits(X, y) - 1
        # here, the scores are normalized by get_n_splits(X, y)
        self.grid_scores_ = scores[::-1] / cv.get_n_splits(X, y)
        return self
        
if __name__ == "__main__":
    from sklearn import svm, datasets
    iris = datasets.load_iris()
    parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
    svr = svm.SVC()
    forward_search = ForwardSearch(svr, parameters)
    forward_search.fit(iris.data, iris.target)
        