import numpy as np
from scipy.spatial import distance_matrix


class SMOTE(object):
    """
    Custom implementation of SMOTE (Chawla et al., 2002).
    Leverages vectorization to interpolate extremely quickly.
    SMOTE generates synthetic examples by way of a linear interpolation
    between a positive-class observation (the "celebrity") and another
    positive-class observation (the "fan") randomly chosen out of `k`
    nearest fans.

    More details here: https://arxiv.org/pdf/1106.1813.pdf
    """

    def __init__(self, k=5, categorical_nunique_thres=5):
        self.k = 5
        self.categorical_nunique_thres = categorical_nunique_thres

    def generate(self,
                 X,
                 y=None,
                 num_to_generate=1,
                 infer_cols=True,
                 smote_cols=None,
                 match_cols=None,):
        """
        Args:
            X (np.ndarray or pd.core.frame.DataFrame): matrix to generate
                synthetic samples from.
            y (np.ndarray or pd.core.series.Series): y-vector containing
                binary class labels.
            infer_columns (bool): whether to infer categorical and
                non-categorical columns.
            match_columns (list): non-continuous columns not to interpolate.
            smote_columns (list): continous columns for interpolation.
        Returns:
            `final_synthetics` (np.ndarray): a dataframe of synthetic samples.
        """
        if not infer_cols and not all([smote_cols, match_cols]):
            raise ValueError("""Either `infer_cols` is True or both
                             `match_cols` and `smote_cols` are specified.""")
        if y is not None:
            X_sm = X[y == 1]
        else:
            X_sm = X

        # separate data to interpolate between and to match on.
        smote_data, categorical_data = self._prepare_data(X_sm)
        match_data = np.hstack((smote_data, categorical_data))
        synthetics = []

        # choose celebrities and one of their `k` closest fans.
        D = self._create_distance_matrix(match_data)
        celeb_indices = sorted(np.random.choice(range(len(match_data)),
                                                num_to_generate))
        celeb_data = smote_data[celeb_indices]
        celeb_cat_data = categorical_data[celeb_indices]
        celeb_distances = D[celeb_indices]
        fan_indices = celeb_distances[np.arange(len(celeb_distances)),
                                      np.random.choice(range(5),
                                                       num_to_generate)]
        fan_data = smote_data[fan_indices]

        # select coefficients & generate synthetics
        coefficients = np.random.uniform(size=(num_to_generate,
                                               smote_data.shape[1]))
        synthetics = celeb_data + np.multiply(coefficients,
                                              celeb_data - fan_data)
        final_synthetics = np.hstack((synthetics, celeb_cat_data))
        return final_synthetics[:, self.column_order]

    def _prepare_data(self, X):
        smote_columns, cat_columns = self._infer_cols(X)
        smote_data = np.array(X[:, smote_columns])  # d_smote < d_match
        cat_cols = list(set(cat_columns).difference(set(smote_columns)))
        categorical_data = np.array(X[:, cat_cols])
        return smote_data, categorical_data

    def _infer_cols(self, X):
        counts = np.apply_along_axis(lambda x: len(np.unique(x)), 1, X)
        cat_cols = np.where(counts <= self.categorical_nunique_thres)[0]
        non_cat_cols = np.where(counts > self.categorical_nunique_thres)[0]
        self.column_order = np.argsort(list(non_cat_cols) + list(cat_cols))
        return non_cat_cols, cat_cols

    def _create_distance_matrix(self, X):
        return np.argsort(distance_matrix(X, X))[:, 1:self.k + 1]