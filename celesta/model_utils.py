"""
model_utils.py — Celesta

Contains BalancedXGBClassifier, the only class shared between train_model.py
and app.py.  It must live in this module (not in either caller) so that joblib
can resolve the class during unpickling regardless of which script loads the
model file.
"""

from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


class BalancedXGBClassifier(XGBClassifier):
    """XGBClassifier with automatic balanced sample weights.

    XGBoost does not honour scikit-learn's ``class_weight`` parameter the way
    tree-based estimators do.  scikit-learn 1.9 also tightened metadata-routing
    rules, making it impossible to pass ``sample_weight`` through
    ``VotingClassifier.fit()`` without a custom router.

    This subclass sidesteps both problems: it computes balanced sample weights
    from ``y`` and injects them before delegating to ``XGBClassifier.fit()``.
    The caller never has to think about class imbalance.

    Parameters
    ----------
    Same as ``xgboost.XGBClassifier``.
    """

    def fit(self, X, y, **kwargs):
        """Fit with balanced sample weights derived from class frequencies.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Integer-encoded class labels.
        **kwargs
            Forwarded to ``XGBClassifier.fit()``.  If ``sample_weight`` is
            already present in ``kwargs`` it is left unchanged.

        Returns
        -------
        self
        """
        kwargs.setdefault("sample_weight", compute_sample_weight("balanced", y))
        return super().fit(X, y, **kwargs)
