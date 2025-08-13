"""
Risk-Controlling Predictor
"""

import numpy as np
import pandas as pd
from scipy.optimize import brentq


from sklearn.base import BaseEstimator, MetaEstimatorMixin
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from random import uniform

from core.fairness_metrics import demographic_parity, equal_opportunity, predictive_equality
from scipy.stats import binom

def ucb_wsr(x, delta, maxiters=1000, B=1, eps=1e-10):
    """
    Compute the upper confidence bound (UCB) based on the Waudby-Smith Ramdas (WSR) bound.

    Args:
        TODO

    Returns:
        TODO
    """
    n = x.shape[0]
    muhat = (np.cumsum(x) + 0.5) / (1 + np.array(range(1, n + 1)))
    sigma2hat = (np.cumsum((x - muhat) ** 2) + 0.25) / (1 + np.array(range(1, n + 1)))
    sigma2hat[1:] = sigma2hat[:-1]
    sigma2hat[0] = 0.25
    nu = np.minimum(np.sqrt(2 * np.log(1 / delta) / n / sigma2hat), 1 / B)

    def _Kn(mu):
        return np.max(np.cumsum(np.log(1 - nu * (x - mu)))) + np.log(delta)

    if _Kn(1) < 0:
        return B
    if _Kn(eps) > 0:
        return eps
    return brentq(_Kn, eps, 1 - eps, maxiter=maxiters)
 
def binomial_p_value(
    risk,
    n,
    alpha=0.05
):
    """
    Compute the binomial lower tail distribution.

    Args:
        risk: Computed risk estimate.
        n: Number of calibration samples.
        alpha: Tolerated risk level.

    Returns:
        p-value.
    """
    p_value = binom.cdf(n * risk, n, alpha)
    return p_value

def ucb_binomial(risk, delta, n_cal, step=0.01, upper=True):
    """
    Compute the upper confidence bound (UCB) based on the binomial distribution.

    Args:
        TODO

    Returns:
        TODO
    """
    if upper==True:
        alphas = np.arange(0.01, 1.0 + step, step)[::-1]
        for i in range(len(alphas)):
            if (
                binomial_p_value(risk=risk, n=n_cal, alpha=alphas[i])
                >= delta
            ):
                return alphas[i]
        return 0.0
    else:
        alphas = np.arange(0.01, 1.0 + step, step)
        for i in range(len(alphas)):
            if (
                1 - binomial_p_value(risk=risk, n=n_cal, alpha=alphas[i])
                >= delta
            ):
                return alphas[i]
        return 1.0



class MonotoneThresholdOptimizer(BaseEstimator, MetaEstimatorMixin):
    """
    For a given score-based predictor, this post-processing mechanism
    computes group-wise thresholds for each group in the sensitive attribute,
    in order to optimize a given fairness notion.
    
    Parameters
    ----------
    estimator : object
        A score-based predictor implementing the scikit-learn estimator interface.
        <https://scikit-learn.org/stable/developers/develop.html#estimators>`_
        whose output is post-processed.
    
    """
    
    def __init__(self, estimator, theta=0.5):
        self.estimator = estimator
        self.lambda_ = 0
        self.theta = theta
        self.fairness_notion = None
        self.utility_metric = None
        
    def _check_fairness_notion(self, fairness_notion):
        if fairness_notion not in ["demographic_parity", "equal_opportunity", "predictive_equality"]:
            raise ValueError(f"Unsupported fairness notion: {fairness_notion}")
        
    def fit(self, X, y, **kwargs):
        """
        Fit the underlying estimator.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        
        y : array-like of shape (n_samples,)
            The target values.
            
        kwargs : dict
            Additional keyword arguments to pass to the underlying estimator's fit method.
        

        Returns
        -------
        self : object
            Returns self.
        """
        
        self.estimator.fit(X, y, **kwargs)
        return self
    
    def predict(self, X, **kwargs):
        """
        Predict the target values.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
            
        kwargs : dict
            Additional keyword arguments to pass to the underlying estimator's predict method.
        

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            The predicted target values.
        """
        
        if not hasattr(self, 'estimator'):
            raise NotFittedError("The estimator has not been fitted yet.")
        
        scores = self.estimator.predict_proba(X)[:, 1]
        y_pred = (scores >= self.theta).astype(int)
        return scores, y_pred
    
    def find_naive_lambda(self, X, s, y, fairness_notion, utility_metric, alpha, step_size=.001, **kwargs):
        """
        Find naive lambda using a validation set using Eq. X. 
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        
        s : array-like of shape (n_samples,)
            The sensitive attribute values.
        
        y : array-like of shape (n_samples,)
            The target values.
            
        fairness_notion : str
            The fairness notion to optimize. Supported values are 'demographic parity',
            'equal opportunity', and 'predictive equality'.
            
        utility_metric : str
            The utility metric to optimize. 
            
        alpha : float
            The tolerance level.
            
        step_size : float
            The step size for the search.
            
        kwargs : dict
            Additional keyword arguments to pass to the underlying estimator's predict method.
        
        
        Returns
        -------
        self : object
            Returns self.
        """
        
        if not hasattr(self, 'estimator'):
            raise NotFittedError("The estimator has not been fitted yet.")
        
        self._check_fairness_notion(fairness_notion)
        self.fairness_notion = fairness_notion
        self.lambda_ = 0.0
        
        # Compute scores and predictions using theta
        scores = self.estimator.predict_proba(X)[:, 1]
        
        scores0 = scores[s == 0]
        scores1 = scores[s == 1]
        y_pred0 = (scores0 >= self.theta).astype(int)
        y_pred1 = (scores1 >= self.theta).astype(int)
        
        y0 = y[s == 0]
        y1 = y[s == 1]
         
        if self.fairness_notion == "demographic_parity":
            r0 = np.mean(y_pred0)
            r1 = np.mean(y_pred1)
        elif self.fairness_notion == "equal_opportunity":
            r0 = 1 - recall_score(y0, y_pred0)
            r1 = 1 - recall_score(y1, y_pred1)
        else:
            r0 = 1 - recall_score(y0, y_pred0, pos_label=0)
            r1 = 1 - recall_score(y1, y_pred1, pos_label=0)
                        
        if r0 < r1:
            self.priv, self.unpriv = 0, 1
            r_priv, r_unpriv = r0, r1
        else:
            self.priv, self.unpriv = 1, 0
            r_priv, r_unpriv = r1, r0
            
        self.theta_priv = self.theta
        self.theta_unpriv = self.theta
        risk = r_unpriv - r_priv

        y_priv = y[s == self.priv]
        y_unpriv = y[s == self.unpriv]
        scores_priv = scores[s == self.priv]
        scores_unpriv = scores[s == self.unpriv]
                        
        while risk > alpha:
            self.lambda_ += step_size
            
            # Update thresholds 
            if self.fairness_notion in ["demographic_parity", "predictive_equality"]:
                self.theta_priv -= step_size
                self.theta_unpriv += step_size
            else:
                self.theta_priv += step_size
                self.theta_unpriv -= step_size
            
            # Compute predictions using updated thresholds
            y_pred_priv = (scores_priv >= self.theta_priv).astype(int)
            y_pred_unpriv = (scores_unpriv >= self.theta_unpriv).astype(int)
            
            if self.fairness_notion == "demographic_parity":
                r_priv = np.mean(y_pred_priv)
                r_unpriv = np.mean(y_pred_unpriv)
            elif self.fairness_notion == "equal_opportunity":
                r_priv = 1 - recall_score(y_priv, y_pred_priv)
                r_unpriv = 1 - recall_score(y_unpriv, y_pred_unpriv)
            else:
                r_priv = 1 - recall_score(y_priv, y_pred_priv, pos_label=0)
                r_unpriv = 1 - recall_score(y_unpriv, y_pred_unpriv, pos_label=0)
                        
            risk = r_unpriv - r_priv
                    
        return self

    def find_crc_lambda(self, X, s, y, fairness_notion, utility_metric, alpha, step_size=.01, **kwargs):
        """
        Find lambda using conformal risk control. 
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        
        s : array-like of shape (n_samples,)
            The sensitive attribute values.
        
        y : array-like of shape (n_samples,)
            The target values.
            
        alpha : float
            The tolerance level.
            
        step_size : float
            The step size for the search.
            
        kwargs : dict
            Additional keyword arguments to pass to the underlying estimator's predict method.
        
        
        Returns
        -------
        self : object
            Returns self.
        """
        
        if not hasattr(self, 'estimator'):
            raise NotFittedError("The estimator has not been fitted yet.")
        
        self._check_fairness_notion(fairness_notion)
        self.fairness_notion = fairness_notion
        self.lambda_ = 0
        
        # Compute scores and predictions using theta
        scores = self.estimator.predict_proba(X)[:, 1]
        
        scores0 = scores[s == 0]
        scores1 = scores[s == 1]
        y_pred0 = (scores0 >= self.theta).astype(int)
        y_pred1 = (scores1 >= self.theta).astype(int)
        
        y0 = y[s == 0]
        y1 = y[s == 1]
         
        if self.fairness_notion == "demographic_parity":
            r0 = np.mean(y_pred0)
            r1 = np.mean(y_pred1)
        elif self.fairness_notion == "equal_opportunity":
            r0 = 1 - recall_score(y0, y_pred0)
            r1 = 1 - recall_score(y1, y_pred1)
        else:
            r0 = 1 - recall_score(y0, y_pred0, pos_label=0)
            r1 = 1 - recall_score(y1, y_pred1, pos_label=0)
                        
        if r0 < r1:
            self.priv, self.unpriv = 0, 1
            r_priv, r_unpriv = r0, r1
        else:
            self.priv, self.unpriv = 1, 0
            r_priv, r_unpriv = r1, r0
            
        self.theta_priv = self.theta
        self.theta_unpriv = self.theta
        risk = r_unpriv - r_priv

        y_priv = y[s == self.priv]
        y_unpriv = y[s == self.unpriv]
        scores_priv = scores[s == self.priv]
        scores_unpriv = scores[s == self.unpriv]
                        
        while risk > alpha - ((1 - alpha) / len(y)):
            self.lambda_ += step_size
            
            # Update thresholds 
            if self.fairness_notion in ["demographic_parity", "predictive_equality"]:
                self.theta_priv -= step_size
                self.theta_unpriv += step_size
            else:
                self.theta_priv += step_size
                self.theta_unpriv -= step_size
            
            # Compute predictions using updated thresholds
            y_pred_priv = (scores_priv >= self.theta_priv).astype(int)
            y_pred_unpriv = (scores_unpriv >= self.theta_unpriv).astype(int)
            
            if self.fairness_notion == "demographic_parity":
                r_priv = np.mean(y_pred_priv)
                r_unpriv = np.mean(y_pred_unpriv)
            elif self.fairness_notion == "equal_opportunity":
                r_priv = 1 - recall_score(y_priv, y_pred_priv)
                r_unpriv = 1 - recall_score(y_unpriv, y_pred_unpriv)
            else:
                r_priv = 1 - recall_score(y_priv, y_pred_priv, pos_label=0)
                r_unpriv = 1 - recall_score(y_unpriv, y_pred_unpriv, pos_label=0)
                        
            risk = r_unpriv - r_priv
                    
        return self

    def find_ucb_lambda(self, X, s, y, fairness_notion, utility_metric, alpha, delta, step_size=.01, bound="CP", **kwargs):
        """
        Find naive lambda using a valid upper confidence bound. 
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        
        s : array-like of shape (n_samples,)
            The sensitive attribute values.
        
        y : array-like of shape (n_samples,)
            The target values.
        
        fairness_notion : str
            The fairness notion to optimize. Supported values are 'demographic parity',
            'equal opportunity', and 'predictive equality'.
            
        utility_metric : str
            The utility metric to optimize.
            
        alpha : float
            The tolerance level.
            
        step_size : float
            The step size for the search.
            
        kwargs : dict
            Additional keyword arguments to pass to the underlying estimator's predict method.
        
        
        Returns
        -------
        self : object
            Returns self.
        """
        
        if not hasattr(self, 'estimator'):
            raise NotFittedError("The estimator has not been fitted yet.")
        
        self._check_fairness_notion(fairness_notion)
        self.fairness_notion = fairness_notion
        self.lambda_ = 0.00

        
        # Compute scores and predictions using theta
        scores = self.estimator.predict_proba(X)[:, 1]
        
        scores0 = scores[s == 0]
        scores1 = scores[s == 1]
        y_pred0 = (scores0 >= self.theta).astype(int)
        y_pred1 = (scores1 >= self.theta).astype(int)
        
        y0 = y[s == 0]
        y1 = y[s == 1]
         
        if self.fairness_notion == "demographic_parity":
            r0 = np.mean(y_pred0)
            r1 = np.mean(y_pred1)
        elif self.fairness_notion == "equal_opportunity":
            r0 = 1 - recall_score(y0, y_pred0)
            r1 = 1 - recall_score(y1, y_pred1)
        else:
            r0 = 1 - recall_score(y0, y_pred0, pos_label=0)
            r1 = 1 - recall_score(y1, y_pred1, pos_label=0)
                        
        if r0 < r1:
            self.priv, self.unpriv = 0, 1
            r_priv, r_unpriv = r0, r1
        else:
            self.priv, self.unpriv = 1, 0
            r_priv, r_unpriv = r1, r0
            
            
        self.theta_priv = self.theta
        self.theta_unpriv = self.theta
        risk = r_unpriv - r_priv

        y_priv = y[s == self.priv]
        y_unpriv = y[s == self.unpriv]
        scores_priv = scores[s == self.priv]
        scores_unpriv = scores[s == self.unpriv]
        
        # Compute predictions using updated thresholds
        y_pred_priv = (scores_priv >= self.theta_priv).astype(int)
        y_pred_unpriv = (scores_unpriv >= self.theta_unpriv).astype(int)
        
        n = len(y_priv)
        
        if fairness_notion == "demographic_parity":
            n = len(y_priv)    
            delta_L_array = y_pred_unpriv - y_pred_priv
            
        elif fairness_notion != "demographic_parity":
            y_unpriv = np.array(y_unpriv)
            y_priv = np.array(y_priv)
            loss_unpriv = (y_unpriv != y_pred_unpriv).astype(int)
            loss_priv = (y_priv != y_pred_priv).astype(int)
            delta_L_array = loss_unpriv - loss_priv
            
        if bound == "CP":     
            L_plus = np.maximum(0, delta_L_array)
            L_minus = np.maximum(0, -delta_L_array)
            p_plus = np.mean(L_plus)
            p_minus = np.mean(L_minus)
        
            p_plus_bound = ucb_binomial(p_plus, delta, n, upper=True)
            p_minus_bound = ucb_binomial(p_minus, 1-delta, n, upper=True)
            ucb = p_plus_bound - p_minus_bound
            print(ucb)
            
        if bound == "WSR":
            ucb = ucb_wsr(delta_L_array, delta)
        
        
         
        #print("P+: ", p_plus)
        #print("P-: ", p_minus)
        
        #p_plus_bound = binom.ppf(2*delta, n, p_plus) / n
        #p_minus_bound = binom.ppf(2*delta, n, p_minus) / n
        
        #print("P+ bound: ", p_plus_bound)
        #print("P- bound: ", p_minus_bound)
        
        #ucb = p_plus_bound - p_minus_bound
        
        #ucb = binom.cdf(np.ceil(n * p_plus), n, 2*delta) #- binom.cdf(np.floor(n * p_minus), n, 2*delta)
        #print(binom.cdf(np.ceil(n * p_plus), n, 2*delta))
        #print(binom.cdf(np.floor(n * p_minus), n, 2*delta))
                
        while ucb > alpha:
            print(self.lambda_)
            self.lambda_ += step_size
            
            # Update thresholds 
            if self.fairness_notion in ["demographic_parity", "predictive_equality"]:
                self.theta_priv -= step_size
                self.theta_unpriv += step_size
            else:
                self.theta_priv += step_size
                self.theta_unpriv -= step_size
            
            # Compute predictions using updated thresholds
            y_pred_priv = (scores_priv >= self.theta_priv).astype(int)
            y_pred_unpriv = (scores_unpriv >= self.theta_unpriv).astype(int)
            
        
            if fairness_notion == "demographic_parity":
                n = len(y_priv)    
                delta_L_array = y_pred_unpriv - y_pred_priv
            
            elif fairness_notion != "demographic_parity":
                y_unpriv = np.array(y_unpriv)
                y_priv = np.array(y_priv)
                loss_unpriv = (y_unpriv != y_pred_unpriv).astype(int)
                loss_priv = (y_priv != y_pred_priv).astype(int)
                delta_L_array = loss_unpriv - loss_priv   
                        
            if bound == "CP":
                Lplus = np.maximum(0, delta_L_array)
                Lminus = np.maximum(0, -delta_L_array)
        
                p_plus = np.mean(Lplus)
                p_minus = np.mean(Lminus)
            
                p_plus_bound = ucb_binomial(p_plus, delta, n, upper=True)
                p_minus_bound = ucb_binomial(p_minus, 1-delta, n, upper=True)
                ucb = p_plus_bound - p_minus_bound
                print(ucb)
                
            if bound == "WSR":
                ucb = ucb_wsr(delta_L_array, delta)
            
            #p_plus_pvalue = binom.cdf(p_plus, n, 2*delta) / n
            #p_minus_bound = binom.ppf(alpha, n, 2*delta) / n
        
            #print("BOUNDS DIF: {}".format(ucb))

            #ucb = binom.cdf(np.ceil(n * r_unpriv), n, 2*delta) #- binom.cdf(np.floor(n * p_minus), n, 2*delta)
            #print(r_unpriv)
            #upc = ucb_hb(risk, delta, len(y), binary_loss=False)

        #print(p_plus)
        #print(p_minus)
        #print(p_plus - p_minus)
            print(f"UCB lambda: {self.lambda_}")
            print(f"Risk for UCB lambda: {risk}")
        
        return self
    
    def evaluate(self, X, s, y, **kwargs):
        scores = self.estimator.predict_proba(X)[:, 1]
        
        scores_priv = scores[s == self.priv]
        scores_unpriv = scores[s == self.unpriv]
        
        y_pred_priv = (scores_priv >= self.theta_priv).astype(int)
        y_pred_unpriv = (scores_unpriv >= self.theta_unpriv).astype(int)
        y_priv = y[s == self.priv]
        y_unpriv = y[s == self.unpriv]

        if self.fairness_notion == "demographic_parity":
            r_priv = np.mean(y_pred_priv)
            r_unpriv = np.mean(y_pred_unpriv)
        elif self.fairness_notion == "equal_opportunity":
            r_priv = 1 - recall_score(y_priv, y_pred_priv)
            r_unpriv = 1 - recall_score(y_unpriv, y_pred_unpriv)
        else:
            r_priv = 1 - recall_score(y_priv, y_pred_priv, pos_label=0)
            r_unpriv = 1 - recall_score(y_unpriv, y_pred_unpriv, pos_label=0)
                        
        unfairness = r_unpriv - r_priv 
        
        y_pred = np.concatenate([y_pred_priv, y_pred_unpriv])
        y_true = np.concatenate([y_priv, y_unpriv])
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        specificity = recall_score(y_true, y_pred, pos_label=0)
        f1 = f1_score(y_true, y_pred)
        
        
        return accuracy, unfairness, precision, recall, specificity, f1

class MonotoneIndividualFlipper(BaseEstimator, MetaEstimatorMixin):
    """
    For a given score-based predictor, this post-processing mechanism
    computes group-wise thresholds for each group in the sensitive attribute,
    in order to optimize a given fairness notion.
    
    Parameters
    ----------
    estimator : object
        A score-based predictor implementing the scikit-learn estimator interface.
        <https://scikit-learn.org/stable/developers/develop.html#estimators>`_
        whose output is post-processed.
    
    """
    
    def __init__(self, estimator, theta=0.5):
        self.estimator = estimator
        self.lambda_ = 0
        self.theta = theta
        self.fairness_notion = None
        self.utility_metric = None
        
    def _check_fairness_notion(self, fairness_notion):
        if fairness_notion not in ["demographic_parity", "equal_opportunity", "predictive_equality"]:
            raise ValueError(f"Unsupported fairness notion: {fairness_notion}")
        
    def fit(self, X, y, **kwargs):
        """
        Fit the underlying estimator.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        
        y : array-like of shape (n_samples,)
            The target values.
            
        kwargs : dict
            Additional keyword arguments to pass to the underlying estimator's fit method.
        

        Returns
        -------
        self : object
            Returns self.
        """
        
        self.estimator.fit(X, y, **kwargs)
        return self
    
    def predict(self, X, **kwargs):
        """
        Predict the target values.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
            
        kwargs : dict
            Additional keyword arguments to pass to the underlying estimator's predict method.
        

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            The predicted target values.
        """
        
        if not hasattr(self, 'estimator'):
            raise NotFittedError("The estimator has not been fitted yet.")
        
        scores = self.estimator.predict_proba(X)[:, 1]
        y_pred = (scores >= self.theta).astype(int)
        return scores, y_pred
    
    def find_naive_lambda(self, X, s, y, fairness_notion, utility_metric, alpha, step_size=.001, **kwargs):
        """
        Find naive lambda using a validation set using Eq. X. 
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        
        s : array-like of shape (n_samples,)
            The sensitive attribute values.
        
        y : array-like of shape (n_samples,)
            The target values.
            
        fairness_notion : str
            The fairness notion to optimize. Supported values are 'demographic parity',
            'equal opportunity', and 'predictive equality'.
            
        utility_metric : str
            The utility metric to optimize. 
            
        alpha : float
            The tolerance level.
            
        step_size : float
            The step size for the search.
            
        kwargs : dict
            Additional keyword arguments to pass to the underlying estimator's predict method.
        
        
        Returns
        -------
        self : object
            Returns self.
        """
        
        if not hasattr(self, 'estimator'):
            raise NotFittedError("The estimator has not been fitted yet.")
        
        self._check_fairness_notion(fairness_notion)
        self.fairness_notion = fairness_notion
        self.lambda_ = 0
        
        # Compute scores and predictions using theta
        scores = self.estimator.predict_proba(X)[:, 1]
        
        scores0 = scores[s == 0]
        scores1 = scores[s == 1]
        y_pred0 = (scores0 >= self.theta).astype(int)
        y_pred1 = (scores1 >= self.theta).astype(int)
        
        y0 = y[s == 0]
        y1 = y[s == 1]
         
        if self.fairness_notion == "demographic_parity":
            r0 = np.mean(y_pred0)
            r1 = np.mean(y_pred1)
        elif self.fairness_notion == "equal_opportunity":
            r0 = 1 - recall_score(y0, y_pred0)
            r1 = 1 - recall_score(y1, y_pred1)
        else:
            r0 = 1 - recall_score(y0, y_pred0, pos_label=0)
            r1 = 1 - recall_score(y1, y_pred1, pos_label=0)
                        
        if r0 < r1:
            self.priv, self.unpriv = 0, 1
            r_priv, r_unpriv = r0, r1
        else:
            self.priv, self.unpriv = 1, 0
            r_priv, r_unpriv = r1, r0
            
        self.theta_priv = self.theta
        self.theta_unpriv = self.theta
        risk = r_unpriv - r_priv

        y_priv = y[s == self.priv]
        y_unpriv = y[s == self.unpriv]
        scores_priv = scores[s == self.priv]
        scores_unpriv = scores[s == self.unpriv]
                        
        while risk > alpha:
            self.lambda_ += step_size
            
            # Update thresholds 
            if self.fairness_notion in ["demographic_parity", "predictive_equality"]:
                self.theta_priv -= step_size
                self.theta_unpriv += step_size
            else:
                self.theta_priv += step_size
                self.theta_unpriv -= step_size
            
            # Compute predictions using updated thresholds
            y_pred_priv = (scores_priv >= self.theta_priv).astype(int)
            y_pred_unpriv = (scores_unpriv >= self.theta_unpriv).astype(int)
            
            if self.fairness_notion == "demographic_parity":
                r_priv = np.mean(y_pred_priv)
                r_unpriv = np.mean(y_pred_unpriv)
            elif self.fairness_notion == "equal_opportunity":
                r_priv = 1 - recall_score(y_priv, y_pred_priv)
                r_unpriv = 1 - recall_score(y_unpriv, y_pred_unpriv)
            else:
                r_priv = 1 - recall_score(y_priv, y_pred_priv, pos_label=0)
                r_unpriv = 1 - recall_score(y_unpriv, y_pred_unpriv, pos_label=0)
                        
            risk = r_unpriv - r_priv
                    
        return self

    def find_crc_lambda(self, X, s, y, fairness_notion, utility_metric, alpha, **kwargs):
        """
        Find lambda using conformal risk control with threshold-aware adaptive step size.
        """

        if not hasattr(self, 'estimator'):
            raise NotFittedError("The estimator has not been fitted yet.")

        self._check_fairness_notion(fairness_notion)
        self.fairness_notion = fairness_notion
        self.lambda_ = 0.0

        # Adaptive step size parameters
        initial_step = 0.01
        min_step = 0.0001
        threshold_ratio = 0.5
        step_size = initial_step

        scores = self.estimator.predict_proba(X)[:, 1]
        y_pred = (scores >= self.theta).astype(int)

        y_pred0 = y_pred[s == 0]
        y_pred1 = y_pred[s == 1]
        y0 = y[s == 0]
        y1 = y[s == 1]

        if self.fairness_notion == "demographic_parity":
            r0, r1 = np.mean(y_pred0), np.mean(y_pred1)
        elif self.fairness_notion == "equal_opportunity":
            r0 = 1 - recall_score(y0, y_pred0)
            r1 = 1 - recall_score(y1, y_pred1)
        else:
            r0 = 1 - recall_score(y0, y_pred0, pos_label=0)
            r1 = 1 - recall_score(y1, y_pred1, pos_label=0)

        if r0 < r1:
            self.priv, self.unpriv = 0, 1
            r_priv, r_unpriv = r0, r1
        else:
            self.priv, self.unpriv = 1, 0
            r_priv, r_unpriv = r1, r0

        risk = r_unpriv - r_priv
        initial_risk = risk
        margin = alpha - ((1 - alpha) / len(y))

        y_priv = y[s == self.priv]
        y_unpriv = y[s == self.unpriv]
        group_mask_priv = s == self.priv
        group_mask_unpriv = s == self.unpriv

        # Precompute once
        X_flipped = X.copy()
        X_flipped["race"] = 1 - X_flipped["race"]
        scores_flipped = self.estimator.predict_proba(X_flipped)[:, 1]
        individual_bias = 1 - np.abs(scores - scores_flipped)

        while risk > margin:
            self.lambda_ += step_size

            if risk < threshold_ratio * initial_risk and step_size > min_step:
                step_size = max(min_step, step_size * 0.5)
            step_size = 0.01

            low_bias_mask = individual_bias < self.lambda_
            priv_unc_mask = low_bias_mask & group_mask_priv
            unpriv_unc_mask = low_bias_mask & group_mask_unpriv

            y_pred_updated = y_pred.copy()
            if self.fairness_notion in ["demographic_parity", "predictive_equality"]:
                y_pred_updated[priv_unc_mask] = 1
                y_pred_updated[unpriv_unc_mask] = 0
            else:
                y_pred_updated[priv_unc_mask] = 0
                y_pred_updated[unpriv_unc_mask] = 1

            y_pred_priv = y_pred_updated[group_mask_priv]
            y_pred_unpriv = y_pred_updated[group_mask_unpriv]

            if self.fairness_notion == "demographic_parity":
                r_priv = np.mean(y_pred_priv)
                r_unpriv = np.mean(y_pred_unpriv)
            elif self.fairness_notion == "equal_opportunity":
                r_priv = 1 - recall_score(y_priv, y_pred_priv)
                r_unpriv = 1 - recall_score(y_unpriv, y_pred_unpriv)
            else:
                r_priv = 1 - recall_score(y_priv, y_pred_priv, pos_label=0)
                r_unpriv = 1 - recall_score(y_unpriv, y_pred_unpriv, pos_label=0)

            risk = r_unpriv - r_priv
            print(f"Lambda for CRC: {self.lambda_}, Risk: {risk}, Step size: {step_size}")

        print(f"Final CRC Lambda: {self.lambda_}, Final Risk: {risk}")
        return self

    def find_ucb_lambda(self, X, s, y, fairness_notion, utility_metric, alpha, delta, step_size=.01, bound="CP", **kwargs):
        """
        Find naive lambda using a valid upper confidence bound. 
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        
        s : array-like of shape (n_samples,)
            The sensitive attribute values.
        
        y : array-like of shape (n_samples,)
            The target values.
        
        fairness_notion : str
            The fairness notion to optimize. Supported values are 'demographic parity',
            'equal opportunity', and 'predictive equality'.
            
        utility_metric : str
            The utility metric to optimize.
            
        alpha : float
            The tolerance level.
            
        step_size : float
            The step size for the search.
            
        kwargs : dict
            Additional keyword arguments to pass to the underlying estimator's predict method.
        
        
        Returns
        -------
        self : object
            Returns self.
        """
        
        if not hasattr(self, 'estimator'):
            raise NotFittedError("The estimator has not been fitted yet.")
        
        self._check_fairness_notion(fairness_notion)
        self.fairness_notion = fairness_notion
        self.lambda_ = 0.6
        
        # Adaptive step size parameters
        initial_step = 0.01
        min_step = 0.001
        threshold_ratio = 0.5
        step_size = initial_step

        # Compute scores and predictions using theta
        scores = self.estimator.predict_proba(X)[:, 1]
        y_pred = (scores >= self.theta).astype(int)
        
        y_pred0 = y_pred[s == 0]
        y_pred1 = y_pred[s == 1]
        y0 = y[s == 0]
        y1 = y[s == 1]
        
        if self.fairness_notion == "demographic_parity":
            r0 = np.mean(y_pred0)
            r1 = np.mean(y_pred1)
        elif self.fairness_notion == "equal_opportunity":
            r0 = 1 - recall_score(y0, y_pred0)
            r1 = 1 - recall_score(y1, y_pred1)
        else:
            r0 = 1 - recall_score(y0, y_pred0, pos_label=0)
            r1 = 1 - recall_score(y1, y_pred1, pos_label=0)
                        
        if r0 < r1:
            self.priv, self.unpriv = 0, 1
            r_priv, r_unpriv = r0, r1
        else:
            self.priv, self.unpriv = 1, 0
            r_priv, r_unpriv = r1, r0
            
        risk = r_unpriv - r_priv
        

        y_priv = y[s == self.priv]
        y_unpriv = y[s == self.unpriv]
        y_pred_priv = y_pred[s == self.priv]
        y_pred_unpriv = y_pred[s == self.unpriv]
        
        n = len(y_priv)
        
        if fairness_notion == "demographic_parity":
            delta_L_array = y_pred_unpriv - y_pred_priv
            
        elif fairness_notion != "demographic_parity":
            y_unpriv = np.array(y_unpriv)
            y_priv = np.array(y_priv)
            loss_unpriv = (y_unpriv != y_pred_unpriv).astype(int)
            loss_priv = (y_priv != y_pred_priv).astype(int)
            delta_L_array = loss_unpriv - loss_priv 
            
        if bound == "CP":
              
        
            L_plus = np.maximum(0, delta_L_array)
            L_minus = np.maximum(0, -delta_L_array)
            p_plus = np.mean(L_plus)
            p_minus = np.mean(L_minus)
        
            p_plus_bound = ucb_binomial(p_plus, delta, n, upper=True)
            p_minus_bound = ucb_binomial(p_minus, 1-delta, n, upper=True)
            ucb = p_plus_bound - p_minus_bound
        
        if bound == "WSR":
            ucb = ucb_wsr(delta_L_array, delta)
            
        
        
        initial_risk = ucb

         
        #print("P+: ", p_plus)
        #print("P-: ", p_minus)
        
        #p_plus_bound = binom.ppf(2*delta, n, p_plus) / n
        #p_minus_bound = binom.ppf(2*delta, n, p_minus) / n
        
        #print("P+ bound: ", p_plus_bound)
        #print("P- bound: ", p_minus_bound)
        
        #ucb = p_plus_bound - p_minus_bound
        
        #ucb = binom.cdf(np.ceil(n * p_plus), n, 2*delta) #- binom.cdf(np.floor(n * p_minus), n, 2*delta)
        #print(binom.cdf(np.ceil(n * p_plus), n, 2*delta))
        #print(binom.cdf(np.floor(n * p_minus), n, 2*delta))
                
        while ucb > alpha:
            #print(ucb)
            self.lambda_ += step_size
            if ucb < threshold_ratio * initial_risk and step_size > min_step:
                step_size = max(min_step, step_size * 0.5)
            
            # Compute scores with senstive attribute flipped
            X_flipped = X.copy()
            X_flipped["race"] = 1 - X_flipped["race"]
            scores_flipped = self.estimator.predict_proba(X_flipped)[:, 1]
            individual_bias = 1 - np.abs(scores - scores_flipped)#+ np.random.uniform(0, 1, size=scores.shape)
            
            unfair_ind = np.where(individual_bias < self.lambda_)[0]
            
            #uncertain_ind = np.where(np.abs(scores - self.theta) < self.lambda_)[0]
            priv_unc_ind = np.where((individual_bias < self.lambda_) & (s==self.priv))[0]
            unpriv_unc_ind = np.where((individual_bias < self.lambda_) & (s==self.unpriv))[0]
            
            #print(len(priv_unc_ind), len(unpriv_unc_ind))
            
            if self.fairness_notion in ["demographic_parity", "predictive_equality"]:
                y_pred[priv_unc_ind] = 1
                y_pred[unpriv_unc_ind] = 0
            else:
                y_pred[priv_unc_ind] = 0
                y_pred[unpriv_unc_ind] = 1
                
            y_pred_priv = y_pred[s == self.priv]
            y_pred_unpriv = y_pred[s == self.unpriv]
            
        
            if fairness_notion == "demographic_parity":
                n = len(y_priv)    
                delta_L_array = y_pred_unpriv - y_pred_priv
            
            elif fairness_notion != "demographic_parity":
                y_unpriv = np.array(y_unpriv)
                y_priv = np.array(y_priv)
                loss_unpriv = (y_unpriv != y_pred_unpriv).astype(int)
                loss_priv = (y_priv != y_pred_priv).astype(int)
                delta_L_array = loss_unpriv - loss_priv
            
            if bound == "CP":
                
                Lplus = np.maximum(0, delta_L_array)
                Lminus = np.maximum(0, -delta_L_array)
        
                p_plus = np.mean(Lplus)
                p_minus = np.mean(Lminus)
            
                p_plus_bound = ucb_binomial(p_plus, delta, n, upper=True)
                p_minus_bound = ucb_binomial(p_minus, 1-delta, n, upper=True)
                ucb = p_plus_bound - p_minus_bound
            
            if bound == "WSR":
                ucb = ucb_wsr(delta_L_array, delta)
            
            
            #p_plus_pvalue = binom.cdf(p_plus, n, 2*delta) / n
            #p_minus_bound = binom.ppf(alpha, n, 2*delta) / n
        

            #ucb = binom.cdf(np.ceil(n * r_unpriv), n, 2*delta) #- binom.cdf(np.floor(n * p_minus), n, 2*delta)
            #print(r_unpriv)
            #upc = ucb_hb(risk, delta, len(y), binary_loss=False)

        #print(p_plus)
        #print(p_minus)
        #print(p_plus - p_minus)
            print(f"UCB lambda: {self.lambda_}")
            print(f"Risk for UCB lambda: {risk}")
        
        return self
    
    def evaluate(self, X, s, y, **kwargs):
        scores = self.estimator.predict_proba(X)[:, 1]
        y_pred = (scores >= self.theta).astype(int)
        
        # Compute scores with senstive attribute flipped
        X_flipped = X.copy()
        X_flipped["race"] = 1 - X_flipped["race"]
        scores_flipped = self.estimator.predict_proba(X_flipped)[:, 1]
        individual_bias = 1 - np.abs(scores - scores_flipped) #+ np.random.uniform(0, 1, size=scores.shape)
            
        unfair_ind = np.where(individual_bias < self.lambda_)[0]
            
        #uncertain_ind = np.where(np.abs(scores - self.theta) < self.lambda_)[0]
        priv_unc_ind = np.where((individual_bias < self.lambda_) & (s==self.priv))[0]
        unpriv_unc_ind = np.where((individual_bias < self.lambda_) & (s==self.unpriv))[0]
        #print(len(priv_unc_ind), len(unpriv_unc_ind))

        
        if self.fairness_notion in ["demographic_parity", "predictive_equality"]:
            y_pred[priv_unc_ind] = 1
            y_pred[unpriv_unc_ind] = 0
        else:
            y_pred[priv_unc_ind] = 0
            y_pred[unpriv_unc_ind] = 1
          
        
        y_pred_priv = y_pred[s == self.priv]
        y_pred_unpriv = y_pred[s == self.unpriv]
        y_priv = y[s == self.priv]
        y_unpriv = y[s == self.unpriv]

        if self.fairness_notion == "demographic_parity":
            r_priv = np.mean(y_pred_priv)
            r_unpriv = np.mean(y_pred_unpriv)
        elif self.fairness_notion == "equal_opportunity":
            r_priv = 1 - recall_score(y_priv, y_pred_priv)
            r_unpriv = 1 - recall_score(y_unpriv, y_pred_unpriv)
        else:
            r_priv = 1 - recall_score(y_priv, y_pred_priv, pos_label=0)
            r_unpriv = 1 - recall_score(y_unpriv, y_pred_unpriv, pos_label=0)
                        
        unfairness = r_unpriv - r_priv 
        

        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        specificity = recall_score(y, y_pred, pos_label=0)
        f1 = f1_score(y, y_pred)
        
        
        return accuracy, unfairness, precision, recall, specificity, f1         
    
  
    
    
def hb_p_value(
    risk: float,
    n: int,
    alpha: float = 0.05,
    eps: float = 1e-3,
    binary_loss: bool = False,
):
    """
    Compute the p-value of the Hoeffding-Bentkus bound.

    Args:
        risk: Computed risk estimate.
        n: Number of calibration samples.
        alpha: Tolerated risk level.

    Returns:
        p-value.
    """
    if binary_loss:
        p_value = binom.cdf(np.ceil(n * risk), n, alpha)
    else:
        bentkus_p_value = np.e * binom.cdf(np.ceil(n * risk), n, alpha)
        a, b = min(risk, alpha), alpha
        h1 = a * np.log(a / b) + (1 - a) * np.log((1 - a) / (1 - b))
        hoeffding_p_value = np.exp(-n * h1)
        p_value = min(bentkus_p_value, hoeffding_p_value)

    assert 0 - eps <= p_value <= 1 + eps, "p-value must be in [0, 1]: {}".format(
        p_value
    )
    return p_value

def ucb_hb(risk, delta, n_cal, binary_loss, step=0.01):
    """
    Compute the upper confidence bound (UCB) based on the Hoeffding-Bentkus (HB) bound.

    Args:
        TODO

    Returns:
        TODO
    """
    alphas = np.arange(0.01, 1.0 + step, step)[::-1]
    for i in range(len(alphas)):
        if (
            hb_p_value(risk=risk, n=n_cal, alpha=alphas[i], binary_loss=binary_loss)
            >= delta
        ):
            return alphas[i]
    return 0.0        
    