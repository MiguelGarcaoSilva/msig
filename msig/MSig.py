import numpy as np
import math
import logging
from typing import Any, Dict, Iterable, List, Sequence, Tuple
from scipy.stats import norm, binom, gaussian_kde, multivariate_normal


logger = logging.getLogger(__name__)


class NullModel:
    """Null model used to estimate pattern probabilities for motifs.

    Parameters
    ----------
    data
        Array-like with shape (m variables, n time points).
    dtypes
        Sequence of Python types for each variable (e.g., int, float, str).
    model
        One of 'empirical', 'kde', or 'gaussian_theoretical'.
    """

    def __init__(self, data: np.ndarray, dtypes: Sequence[type], model: str = "empirical") -> None:
        self.data: np.ndarray = np.asarray(data)
        self.dtypes: Sequence[type] = tuple(dtypes)
        self.model: str = model
        self.pre_computed_distribution: Dict[int, Any] = {}
        self.pre_computed_bivariate_distribution: Dict[int, Any] = {}

        # if any of dtypes is not float, and model is not empirical, raise error
        if any(dtype != float for dtype in dtypes) and model != "empirical":
            raise ValueError("Invalid data type for model %s" % model)

        for var_index, y_j in enumerate(self.data):
            y_j = np.array(y_j, dtype=dtypes[var_index])

            if self.model == "empirical":
                continue

            # pairs for a first-order Markov bivariate distribution
            pairs = np.array([[y_j[i], y_j[i + 1]] for i in range(len(y_j) - 1)]).T
            means = np.mean(pairs[0]), np.mean(pairs[1])

            if self.model == "kde":
                self.pre_computed_distribution[var_index] = gaussian_kde(y_j)
                self.pre_computed_bivariate_distribution[var_index] = gaussian_kde(pairs)
            elif self.model == "gaussian_theoretical":
                self.pre_computed_distribution[var_index] = norm(np.mean(y_j), np.std(y_j))
                self.pre_computed_bivariate_distribution[var_index] = multivariate_normal(means, np.cov(pairs))
            else:
                raise ValueError("Invalid model")

    def vars_indep_time_markov(
        self,
        motif_subsequence: Sequence[np.ndarray],
        variables: Sequence[int],
        delta_thresholds: Sequence[float],
    ) -> float:
        """Estimate the pattern probability assuming variables are independent and time follows a first-order Markov.

        motif_subsequence: sequence of 1D arrays, one per selected variable.
        variables: indices of variables in the original data corresponding to motif_subsequence entries.
        delta_thresholds: per-variable numeric thresholds.
        """
        p_Q: float = 1.0

        # for each subsequence variable
        for seq_idx, subsequence in enumerate(motif_subsequence):
            var_index = variables[seq_idx]
            delta = delta_thresholds[var_index]
            p_Q_j: float = 1.0

            # use dtype that corresponds to the variable index (bugfix: previously used seq_idx)
            dtype = self.dtypes[var_index]
            time_series = np.array(self.data[var_index], dtype=dtype)
            subsequence = np.array(subsequence, dtype=dtype)

            if self.model != "empirical":
                dist = self.pre_computed_distribution[var_index]
                dist_bivar = self.pre_computed_bivariate_distribution[var_index]

            # P(Y_j = x_0^j)
            if delta != 0:
                xi_lower, xi_upper = subsequence[0] - delta, subsequence[0] + delta
            else:
                # define zero-width interval for non-empirical models; empirical handles exact equality below
                xi_lower = xi_upper = subsequence[0]

            if self.model == "empirical":
                if delta != 0:
                    count = np.sum(np.logical_and(time_series >= xi_lower, time_series <= xi_upper))
                else:
                    count = np.sum(time_series == subsequence[0])
                p_Q_j *= count / len(time_series) if len(time_series) > 0 else 0.0
            elif self.model == "kde":
                p_Q_j *= float(dist.integrate_box_1d(xi_lower, xi_upper))
            elif self.model == "gaussian_theoretical":
                p_Q_j *= float(dist.cdf(xi_upper) - dist.cdf(xi_lower))

            # conditional probabilities for subsequent positions in the subsequence
            for i in range(1, len(subsequence)):
                if delta != 0:
                    xi_lower, xi_upper = subsequence[i] - delta, subsequence[i] + delta
                    ximinus1_lower, ximinus1_upper = subsequence[i - 1] - delta, subsequence[i - 1] + delta
                else:
                    xi_lower = xi_upper = subsequence[i]
                    ximinus1_lower = ximinus1_upper = subsequence[i - 1]

                # P(A|B) = P(A ^ B)/P(B)
                if self.model == "empirical":
                    if delta == 0:
                        count = np.sum((time_series[:-1] == subsequence[i - 1]) & (time_series[1:] == subsequence[i]))
                    else:
                        count = np.sum(
                            (np.logical_and(time_series[:-1] >= ximinus1_lower, time_series[:-1] <= ximinus1_upper))
                            & (np.logical_and(time_series[1:] >= xi_lower, time_series[1:] <= xi_upper))
                        )
                    numerator = count / (len(time_series) - 1) if len(time_series) > 1 else 0.0

                    if delta == 0:
                        count = np.sum(time_series == subsequence[i - 1])
                    else:
                        count = np.sum(np.logical_and(time_series >= ximinus1_lower, time_series <= ximinus1_upper))
                    denominator = count / len(time_series) if len(time_series) > 0 else 1.0
                elif self.model == "kde":
                    numerator = float(dist_bivar.integrate_box([ximinus1_lower, xi_lower], [ximinus1_upper, xi_upper]))
                    # use marginal for the previous state as denominator
                    denominator = float(dist.integrate_box_1d(ximinus1_lower, ximinus1_upper))
                elif self.model == "gaussian_theoretical":
                    numerator = float(dist_bivar.cdf([ximinus1_upper, xi_upper]) - dist_bivar.cdf([ximinus1_lower, xi_lower]))
                    denominator = float(dist.cdf(ximinus1_upper) - dist.cdf(ximinus1_lower))

                # avoid division by zero
                cond_p = numerator / denominator if denominator != 0 else 0.0
                p_Q_j *= min(1.0, cond_p)

            logger.debug("subsequence=%s p_Q_j=%E", subsequence, p_Q_j)
            p_Q *= p_Q_j

        return float(p_Q)

    # TODO: Assuming variables are independent and first-order markov between time points
    def vars_dep_time_markov(self, motif_subsequence: Sequence[np.ndarray], variables: Sequence[int]) -> float:
        raise NotImplementedError("Still not implemented")

    # TODO: Assuming variables are independent and independence between time points
    def var_indep_time_indep(self) -> float:
        raise NotImplementedError("Still not implemented")

    @staticmethod
    def hochberg_critical_value(p_values: Iterable[float], false_discovery_rate: float = 0.05) -> float:
        """
        Benjamini-Hochberg FDR correction (standard implementation).
        
        Follows the original Benjamini & Hochberg (1995) procedure and matches
        implementations in R (p.adjust), Python (statsmodels, scipy).
        
        Uses non-strict inequality (≤) as per standard practice.
        """
        ordered_pvalue = sorted(p_values)
        critical_values = [(i / len(ordered_pvalue)) * false_discovery_rate for i in range(1, len(ordered_pvalue) + 1)]
        critical_val = ordered_pvalue[-1]

        for i in reversed(range(len(ordered_pvalue))):
            if ordered_pvalue[i] <= critical_values[i]:  # Standard BH uses ≤
                critical_val = ordered_pvalue[i]
                break

        return min(critical_val, false_discovery_rate)

    @staticmethod
    def bonferonni_correction(p_values: Iterable[float], alpha: float = 0.05) -> float:
        pv_list = list(p_values)
        return alpha / len(pv_list)


class Motif:
    def __init__(
        self,
        multivar_sequence: Sequence[np.ndarray],
        variables: Sequence[int],
        delta_thresholds: Sequence[float],
        n_matches: int,
        pattern_probability: float = 0.0,
        pvalue: float = 1.0,
    ) -> None:
        self.multivar_sequence = multivar_sequence
        self.variables = variables
        self.delta_thresholds = delta_thresholds
        self.n_matches = int(n_matches)
        self.p_Q = float(pattern_probability)
        self.pvalue = float(pvalue)

    def set_pattern_probability(self, model: NullModel, vars_indep: bool = True) -> float:
        self.p_Q = (
            model.vars_indep_time_markov(self.multivar_sequence, self.variables, self.delta_thresholds)
            if vars_indep
            else model.vars_dep_time_markov(self.multivar_sequence, self.variables)
        )
        return self.p_Q

    def set_significance(self, max_possible_matches: int, data_n_variables: int, idd_correction: bool = False) -> float:
        """Compute p-value for observing at least self.n_matches given pattern probability self.p_Q.

        Uses a numerically-stable survival function for the binomial distribution.
        """
        if self.p_Q in [0.0, 1.0]:
            return float(self.p_Q)

        if self.n_matches >= max_possible_matches:
            # undefined/degenerate in original implementation
            return float("nan")

        # tail probability P(X >= n_matches) = sf(n_matches - 1)
        try:
            pvalue = float(binom.sf(self.n_matches - 1, max_possible_matches, self.p_Q))
        except Exception:
            # fallback to manual sum if something unexpected happens
            pvalue = 0.0
            for j in range(self.n_matches, max_possible_matches + 1):
                try:
                    pvalue += float(binom.pmf(j, max_possible_matches, self.p_Q))
                except OverflowError:
                    pvalue += 0.0

        if idd_correction:
            pvalue = min(1.0, pvalue * math.comb(data_n_variables, len(self.variables)))

        self.pvalue = pvalue
        logging.info("p_value = %.3E (p_pattern = %.3E)", self.pvalue, self.p_Q)
        return pvalue
