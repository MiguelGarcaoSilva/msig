import numpy as np
import math
import logging
from collections.abc import Iterable, Sequence
from typing import Any
from scipy.stats import norm, binom, gaussian_kde, multivariate_normal


logger = logging.getLogger(__name__)


def benjamini_hochberg_fdr(p_values: Iterable[float], false_discovery_rate: float = 0.05) -> float:
    """
    Benjamini-Hochberg FDR correction (standard implementation).
    
    Compute the critical value for controlling the False Discovery Rate (FDR)
    using the Benjamini-Hochberg procedure (Benjamini & Hochberg, 1995).
    
    This method controls the expected proportion of false discoveries among
    rejected hypotheses. It is less conservative than Bonferroni correction
    and has greater power for multiple comparisons.
    
    Parameters
    ----------
    p_values : Iterable[float]
        Collection of p-values to correct.
    false_discovery_rate : float, default=0.05
        The desired FDR level (typically 0.05 or 0.01).
        
    Returns
    -------
    float
        The critical p-value threshold. P-values less than or equal to this
        threshold are considered significant after FDR correction.
        
    Notes
    -----
    Follows the original Benjamini & Hochberg (1995) procedure and matches
    implementations in R (p.adjust), Python (statsmodels, scipy).
    Uses non-strict inequality (≤) as per standard practice.
    
    References
    ----------
    Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate:
    a practical and powerful approach to multiple testing. Journal of the Royal
    Statistical Society: Series B, 57(1), 289-300.
    
    Examples
    --------
    >>> p_vals = [0.001, 0.008, 0.039, 0.041, 0.042]
    >>> critical = benjamini_hochberg_fdr(p_vals, false_discovery_rate=0.05)
    >>> significant = [p for p in p_vals if p <= critical]
    """
    ordered_pvalue = sorted(p_values)
    n = len(ordered_pvalue)
    
    if n == 0:
        return 0.0
    
    critical_values = [(i / n) * false_discovery_rate for i in range(1, n + 1)]
    critical_val = ordered_pvalue[-1]

    for i in reversed(range(n)):
        if ordered_pvalue[i] <= critical_values[i]:
            critical_val = ordered_pvalue[i]
            break

    return min(critical_val, false_discovery_rate)


def bonferroni_correction(p_values: Iterable[float], alpha: float = 0.05) -> float:
    """
    Bonferroni correction for multiple hypothesis testing.
    
    Compute the adjusted significance threshold using the conservative
    Bonferroni correction method.
    
    Parameters
    ----------
    p_values : Iterable[float]
        Collection of p-values to correct.
    alpha : float, default=0.05
        The family-wise error rate (FWER) to control.
        
    Returns
    -------
    float
        The corrected significance threshold (alpha / number of tests).
        
    Notes
    -----
    The Bonferroni correction controls the family-wise error rate by dividing
    the significance level by the number of comparisons. It is very conservative
    and may have low power when many comparisons are made.
    
    Examples
    --------
    >>> p_vals = [0.001, 0.008, 0.039, 0.041, 0.042]
    >>> threshold = bonferroni_correction(p_vals, alpha=0.05)
    >>> threshold
    0.01
    """
    pv_list = list(p_values)
    if len(pv_list) == 0:
        return alpha
    return alpha / len(pv_list)


class NullModel:
    """
    Null model for estimating pattern probabilities in multivariate time series motifs.
    
    This class builds a statistical null model from observed data to compute the
    probability that a specific pattern occurs by chance. Three modeling approaches
    are supported: empirical (frequency-based), kernel density estimation (KDE),
    and theoretical Gaussian.

    Parameters
    ----------
    data : array-like of shape (m, n)
        Multivariate time series data where m is the number of variables and
        n is the number of time points. Each row represents one variable's
        time series.
    dtypes : Sequence[type]
        Python types for each variable (e.g., [int, float, str]). Must match
        the number of variables (rows) in data.
    model : {'empirical', 'kde', 'gaussian_theoretical'}, default='empirical'
        The type of null model to use:
        
        - 'empirical': Frequency-based probabilities from observed data.
          Works with any data type (int, float, str).
        - 'kde': Kernel Density Estimation for smooth probability densities.
          Requires all variables to be float type.
        - 'gaussian_theoretical': Assumes Gaussian distribution with observed
          mean and standard deviation. Requires all variables to be float type.
          
    Raises
    ------
    ValueError
        If data shape doesn't match dtypes length.
    ValueError
        If non-float dtypes are used with 'kde' or 'gaussian_theoretical' models.
    ValueError
        If an invalid model type is specified.
    ValueError
        If data contains insufficient time points for the model.
        
    Attributes
    ----------
    data : np.ndarray
        The multivariate time series data.
    dtypes : tuple[type, ...]
        The data types for each variable.
    model : str
        The null model type being used.
    pre_computed_distribution : dict[int, Any]
        Pre-computed marginal distributions for each variable.
    pre_computed_bivariate_distribution : dict[int, Any]
        Pre-computed bivariate distributions for first-order Markov modeling.
        
    Examples
    --------
    >>> import numpy as np
    >>> data = np.random.randn(3, 100)  # 3 variables, 100 time points
    >>> model = NullModel(data, dtypes=[float, float, float], model='empirical')
    
    >>> # For categorical data
    >>> cat_data = np.array([['A', 'B', 'A', 'C'], ['X', 'Y', 'X', 'X']])
    >>> model = NullModel(cat_data, dtypes=[str, str], model='empirical')
    """

    def __init__(self, data: np.ndarray, dtypes: Sequence[type], model: str = "empirical") -> None:
        # Validate inputs
        self.data: np.ndarray = np.asarray(data)
        
        if self.data.ndim != 2:
            raise ValueError(f"Data must be 2-dimensional (variables × time points), got shape {self.data.shape}")
        
        if len(dtypes) != self.data.shape[0]:
            raise ValueError(
                f"Number of dtypes ({len(dtypes)}) must match number of variables "
                f"({self.data.shape[0]})"
            )
        
        if self.data.shape[1] < 2:
            raise ValueError(f"Data must have at least 2 time points, got {self.data.shape[1]}")
        
        if model not in ['empirical', 'kde', 'gaussian_theoretical']:
            raise ValueError(
                f"Invalid model '{model}'. Must be one of: 'empirical', 'kde', 'gaussian_theoretical'"
            )
        
        self.dtypes: tuple[type, ...] = tuple(dtypes)
        self.model: str = model
        self.pre_computed_distribution: dict[int, Any] = {}
        self.pre_computed_bivariate_distribution: dict[int, Any] = {}

        # Validate dtype compatibility with model
        if any(dtype != float for dtype in dtypes) and model != "empirical":
            raise ValueError(
                f"Model '{model}' requires all variables to be float type. "
                f"Use model='empirical' for non-numeric data."
            )

        # Pre-compute distributions for each variable
        for var_index, y_j in enumerate(self.data):
            y_j = np.array(y_j, dtype=dtypes[var_index])

            if self.model == "empirical":
                # Empirical model doesn't need pre-computation
                continue

            # Pairs for a first-order Markov bivariate distribution
            pairs = np.array([[y_j[i], y_j[i + 1]] for i in range(len(y_j) - 1)]).T
            means = np.mean(pairs[0]), np.mean(pairs[1])

            if self.model == "kde":
                self.pre_computed_distribution[var_index] = gaussian_kde(y_j)
                self.pre_computed_bivariate_distribution[var_index] = gaussian_kde(pairs)
            elif self.model == "gaussian_theoretical":
                std_dev = np.std(y_j)
                if std_dev == 0:
                    logger.warning(
                        f"Variable {var_index} has zero standard deviation. "
                        "Using small epsilon for numerical stability."
                    )
                    std_dev = 1e-10
                self.pre_computed_distribution[var_index] = norm(np.mean(y_j), std_dev)
                self.pre_computed_bivariate_distribution[var_index] = multivariate_normal(means, np.cov(pairs))

    def vars_indep_time_markov(
        self,
        motif_subsequence: Sequence[np.ndarray],
        variables: Sequence[int],
        delta_thresholds: Sequence[float],
    ) -> float:
        """
        Estimate pattern probability assuming independent variables and first-order Markov time dependency.
        
        Computes the probability of observing a specific multivariate pattern under the
        assumption that different variables are independent of each other, but temporal
        dependencies within each variable follow a first-order Markov process.
        
        Parameters
        ----------
        motif_subsequence : Sequence[np.ndarray]
            Sequence of 1D arrays, one per selected variable, defining the pattern.
            Each array contains the pattern values for that variable over time.
        variables : Sequence[int]
            Indices of variables in the original data corresponding to each element
            in motif_subsequence. Length must match motif_subsequence.
        delta_thresholds : Sequence[float]
            Per-variable tolerance thresholds for pattern matching. For continuous
            variables, this defines the width of the matching interval. For discrete
            variables, use 0 for exact matching.
            
        Returns
        -------
        float
            The estimated pattern probability P(Q) where 0 ≤ P(Q) ≤ 1.
            Returns 0.0 if the pattern is impossible under the null model.
            
        Notes
        -----
        The probability is computed as:
        
        P(Q) = ∏_{j} P(Q_j)
        
        where P(Q_j) for each variable j is:
        
        P(Q_j) = P(x_0^j) ∏_{t=1}^{L-1} P(x_t^j | x_{t-1}^j)
        
        using the first-order Markov assumption P(x_t | x_{t-1}).
        
        If conditional probabilities exceed 1.0 (due to numerical issues in KDE
        integration), they are clamped to 1.0 with a debug warning.
        
        Examples
        --------
        >>> model = NullModel(data, dtypes=[float, float], model='empirical')
        >>> pattern = [np.array([1.0, 1.1, 1.0]), np.array([2.0, 2.1, 2.0])]
        >>> prob = model.vars_indep_time_markov(pattern, variables=[0, 1], delta_thresholds=[0.2, 0.2])
        """
        p_Q: float = 1.0

        # For each subsequence variable
        for seq_idx, subsequence in enumerate(motif_subsequence):
            var_index = variables[seq_idx]
            delta = delta_thresholds[seq_idx]  # Use seq_idx: delta_thresholds aligns with motif_subsequence
            p_Q_j: float = 1.0

            # Use dtype that corresponds to the variable index in original data
            dtype = self.dtypes[var_index]
            time_series = np.array(self.data[var_index], dtype=dtype)
            subsequence = np.array(subsequence, dtype=dtype)

            if self.model != "empirical":
                dist = self.pre_computed_distribution[var_index]
                dist_bivar = self.pre_computed_bivariate_distribution[var_index]

            # P(Y_j = x_0^j) - Initial probability
            if delta != 0:
                xi_lower, xi_upper = subsequence[0] - delta, subsequence[0] + delta
            else:
                # Zero-width interval for exact matching
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

            # Conditional probabilities for subsequent positions: P(x_t | x_{t-1})
            for i in range(1, len(subsequence)):
                if delta != 0:
                    xi_lower, xi_upper = subsequence[i] - delta, subsequence[i] + delta
                    ximinus1_lower, ximinus1_upper = subsequence[i - 1] - delta, subsequence[i - 1] + delta
                else:
                    xi_lower = xi_upper = subsequence[i]
                    ximinus1_lower = ximinus1_upper = subsequence[i - 1]

                # P(A|B) = P(A ∩ B) / P(B)
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
                    # Use marginal for the previous state as denominator
                    denominator = float(dist.integrate_box_1d(ximinus1_lower, ximinus1_upper))
                elif self.model == "gaussian_theoretical":
                    numerator = float(dist_bivar.cdf([ximinus1_upper, xi_upper]) - dist_bivar.cdf([ximinus1_lower, xi_lower]))
                    denominator = float(dist.cdf(ximinus1_upper) - dist.cdf(ximinus1_lower))

                # Avoid division by zero
                if denominator == 0:
                    cond_p = 0.0
                else:
                    cond_p = numerator / denominator
                    # Warn if probability exceeds 1.0 (numerical issue)
                    if cond_p > 1.0:
                        logger.debug(
                            f"Conditional probability {cond_p:.6f} > 1.0 for variable {var_index}, "
                            f"position {i}. Clamping to 1.0. This may indicate numerical precision issues."
                        )
                        cond_p = 1.0
                
                p_Q_j *= cond_p

            logger.debug("subsequence=%s p_Q_j=%E", subsequence, p_Q_j)
            p_Q *= p_Q_j

        return float(p_Q)

    def vars_dep_time_markov(self, motif_subsequence: Sequence[np.ndarray], variables: Sequence[int]) -> float:
        """
        Estimate pattern probability assuming dependent variables and first-order Markov time dependency.
        
        This method is not yet implemented. Use vars_indep_time_markov() for the current
        implementation which assumes variable independence.
        
        Parameters
        ----------
        motif_subsequence : Sequence[np.ndarray]
            Sequence of 1D arrays defining the pattern.
        variables : Sequence[int]
            Indices of variables in the original data.
            
        Raises
        ------
        NotImplementedError
            This method is not yet implemented.
            
        Notes
        -----
        Future implementation will model dependencies between variables using
        multivariate distributions rather than treating each variable independently.
        """
        raise NotImplementedError(
            "Variable dependency modeling is not yet implemented. "
            "Use vars_indep=True in set_pattern_probability() for the current implementation."
        )


class Motif:
    """
    Represents a multivariate time series motif with statistical significance testing.
    
    A motif is a recurrent pattern in multivariate time series data. This class
    stores the pattern definition, its occurrence count, and statistical measures
    (pattern probability and p-value) computed against a null model.
    
    Parameters
    ----------
    multivar_sequence : Sequence[np.ndarray]
        The motif pattern as a sequence of 1D arrays, one per variable.
        Each array contains the pattern values for that variable over time.
    variables : Sequence[int]
        Indices of the variables in the original data that this motif spans.
        Length must match multivar_sequence.
    delta_thresholds : Sequence[float]
        Per-variable tolerance thresholds for pattern matching. For continuous
        variables, defines the matching interval width. For discrete variables,
        use 0 for exact matching. Length must match multivar_sequence.
    n_matches : int
        Number of times this pattern appears in the data. Must be positive.
    pattern_probability : float, default=0.0
        Probability of observing this pattern under the null model.
        Set to 0.0 initially and computed via set_pattern_probability().
    pvalue : float, default=1.0
        Statistical significance p-value from binomial test.
        Set to 1.0 initially (most conservative) and computed via set_significance().
        
    Attributes
    ----------
    multivar_sequence : Sequence[np.ndarray]
        The motif pattern definition.
    variables : Sequence[int]
        Variable indices for this motif.
    delta_thresholds : Sequence[float]
        Matching tolerance thresholds.
    n_matches : int
        Number of pattern occurrences.
    p_Q : float
        Pattern probability under the null model.
    pvalue : float
        Statistical significance p-value.
        
    Raises
    ------
    ValueError
        If n_matches is not positive.
    ValueError
        If array lengths don't match (multivar_sequence, variables, delta_thresholds).
        
    Examples
    --------
    >>> import numpy as np
    >>> from msig import Motif, NullModel
    >>> 
    >>> # Create a motif representing a pattern found 15 times
    >>> pattern = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])]
    >>> variables = [0, 1]  # First two variables
    >>> thresholds = [0.1, 0.1]  # 0.1 tolerance for both
    >>> motif = Motif(pattern, variables, thresholds, n_matches=15)
    >>> 
    >>> # Compute statistical significance
    >>> data = np.random.randn(2, 1000)
    >>> null_model = NullModel(data, dtypes=[float, float])
    >>> motif.set_pattern_probability(null_model, vars_indep=True)
    >>> motif.set_significance(data_length=1000)
    >>> print(f"p-value: {motif.pvalue:.4f}")
    """
    def __init__(
        self,
        multivar_sequence: Sequence[np.ndarray],
        variables: Sequence[int],
        delta_thresholds: Sequence[float],
        n_matches: int,
        pattern_probability: float = 0.0,
        pvalue: float = 1.0,
    ) -> None:
        # Validate n_matches
        if n_matches <= 0:
            raise ValueError(f"n_matches must be positive, got {n_matches}")
        
        # Validate array lengths match
        if not (len(multivar_sequence) == len(variables) == len(delta_thresholds)):
            raise ValueError(
                f"Length mismatch: multivar_sequence={len(multivar_sequence)}, "
                f"variables={len(variables)}, delta_thresholds={len(delta_thresholds)}. "
                "All must have the same length."
            )
        
        self.multivar_sequence = multivar_sequence
        self.variables = variables
        self.delta_thresholds = delta_thresholds
        self.n_matches = int(n_matches)
        self.p_Q = float(pattern_probability)
        self.pvalue = float(pvalue)

    def set_pattern_probability(self, model: NullModel, vars_indep: bool = True) -> float:
        """
        Compute and set the probability of this pattern under a null model.
        
        Parameters
        ----------
        model : NullModel
            The null model to use for probability estimation.
        vars_indep : bool, default=True
            If True, assumes variables are independent (uses vars_indep_time_markov).
            If False, assumes variables are dependent (not yet implemented).
            
        Returns
        -------
        float
            The computed pattern probability, also stored in self.p_Q.
            
        Raises
        ------
        NotImplementedError
            If vars_indep=False (variable dependency not yet implemented).
            
        Examples
        --------
        >>> motif.set_pattern_probability(null_model, vars_indep=True)
        0.00234
        """
        self.p_Q = (
            model.vars_indep_time_markov(self.multivar_sequence, self.variables, self.delta_thresholds)
            if vars_indep
            else model.vars_dep_time_markov(self.multivar_sequence, self.variables)
        )
        return self.p_Q

    def set_significance(self, max_possible_matches: int, data_n_variables: int, idd_correction: bool = False) -> float:
        """
        Compute statistical significance (p-value) for the observed pattern occurrences.
        
        Tests whether the observed number of matches is statistically significant
        under the null hypothesis using a binomial test. The p-value represents
        P(X >= n_matches | p_Q), where X ~ Binomial(max_possible_matches, p_Q).
        
        Parameters
        ----------
        max_possible_matches : int
            Maximum possible number of pattern occurrences in the data.
            Typically (n_time_points - pattern_length + 1).
        data_n_variables : int
            Total number of variables in the original dataset.
        idd_correction : bool, default=False
            If True, applies IDD (Independent Dimension Discovery) correction by
            adjusting p-value using the Benjamini-Hochberg FDR procedure across
            variable subsets. See Notes for details.
            
        Returns
        -------
        float
            The computed p-value, also stored in self.pvalue.
            Returns 0.0 if pattern_probability is 0.0 (deterministic pattern).
            Returns 1.0 if pattern_probability is 1.0 (completely random).
            Returns NaN if n_matches >= max_possible_matches (degenerate case).
            
        Raises
        ------
        ValueError
            If max_possible_matches <= 0.
        ValueError
            If data_n_variables <= 0.
        OverflowError
            If binomial computation overflows (falls back to manual summation).
            
        Notes
        -----
        The binomial test uses the survival function for numerical stability:
        P(X >= k) = sf(k-1) = 1 - cdf(k-1)
        
        IDD Correction:
        When idd_correction=True, the method adjusts for multiple hypothesis testing
        across different variable subsets. For a motif using k variables from m total,
        there are C(m,k) possible k-variable subsets. 
        
        Examples
        --------
        >>> motif.set_significance(max_possible_matches=1000, data_n_variables=5)
        0.0023
        >>> 
        >>> # With IDD correction for multiple testing
        >>> motif.set_significance(max_possible_matches=1000, data_n_variables=5, idd_correction=True)
        0.0115
        """
        # Validate inputs
        if max_possible_matches <= 0:
            raise ValueError(f"max_possible_matches must be positive, got {max_possible_matches}")
        if data_n_variables <= 0:
            raise ValueError(f"data_n_variables must be positive, got {data_n_variables}")
        
        # Handle edge cases
        if self.p_Q in [0.0, 1.0]:
            return float(self.p_Q)

        if self.n_matches >= max_possible_matches:
            logger.warning(
                f"Degenerate case: n_matches={self.n_matches} >= max_possible_matches={max_possible_matches}. "
                "Returning NaN."
            )
            return float("nan")

        # Compute binomial tail probability P(X >= n_matches)
        try:
            pvalue = float(binom.sf(self.n_matches - 1, max_possible_matches, self.p_Q))
        except OverflowError as e:
            # Fallback to manual sum if binomial computation overflows
            logger.warning(
                f"Binomial computation overflow: {e}. "
                f"Falling back to manual summation for n_matches={self.n_matches}, "
                f"max={max_possible_matches}, p_Q={self.p_Q:.6e}"
            )
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
