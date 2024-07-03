import numpy as np
import math
from scipy.stats import norm, binom, gaussian_kde, multivariate_normal
import logging

logger = logging.getLogger(__name__)

class NullModel:
    data = None
    dtypes = None
    pre_computed_distribution = {}
    pre_computed_bivariate_distribution = {}
    model = None

    def __init__(self, data, dtypes, model="empirical"):
        self.data = data
        self.dtypes = dtypes
        self.model = model

        #if any of dtypes is not float, if model is not empirical, raise error
        if any([dtype != float for dtype in dtypes]) and model != "empirical":
            raise ValueError("Invalid data type for model %s" % model)

        for var_index, y_j in enumerate(data):
            try:
                y_j = np.array(y_j, dtype=dtypes[var_index])
            except ValueError:
                raise ValueError("Invalid data type for variable %d" % var_index)

            if self.model == "empirical":
                continue
    
            pairs = []
            for i in range(len(y_j)):
                if i < len(y_j)-1:
                    pairs.append([y_j[i], y_j[i+1]])            
            pairs = np.array(pairs).T
            means = np.mean(pairs[0]), np.mean(pairs[1])

            if self.model == "kde":
                self.pre_computed_distribution[var_index] = gaussian_kde(y_j)
                self.pre_computed_bivariate_distribution[var_index] = gaussian_kde(pairs)

            elif self.model == "gaussian_theoretical":
                self.pre_computed_distribution[var_index] = norm(np.mean(y_j), np.std(y_j))
                self.pre_computed_bivariate_distribution[var_index] = multivariate_normal(means, np.cov(pairs))
            else:
                raise ValueError("Invalid model")


    def vars_indep_time_markov(self, motif_subsequence, variables, delta_thresholds):
        p_Q = 1
        # for variable j in variables of subsequence J:
        for j, subsequence in enumerate(motif_subsequence):
            var_index = variables[j]
            delta = delta_thresholds[var_index]
            p_Q_j = 1
            time_series = self.data[var_index].astype(self.dtypes[j])
            try:
                subsequence = subsequence.astype(self.dtypes[j])
            except ValueError:
                raise ValueError("Invalid data type for variable %d" % var_index)

            if self.model != "empirical":
                dist = self.pre_computed_distribution[var_index]
                dist_bivar = self.pre_computed_bivariate_distribution[var_index]

            # P(Y_j = x_0^j)
            if delta != 0:
                xi_lower, xi_upper = subsequence[0] - delta, subsequence[0] + delta
            if self.model == "empirical":
                if delta == 0:
                    count = np.sum(time_series == subsequence[0])
                    logging.debug(count)
                else:
                    count = np.sum(np.logical_and(time_series >= xi_lower, time_series <= xi_upper))
                    logging.debug(count)
                p_Q_j *= count / len(time_series)
            elif self.model == "kde":
                p_Q_j *= dist.integrate_box_1d(xi_lower, xi_upper)
            elif self.model == "gaussian_theoretical":
                p_Q_j *= dist.cdf(xi_upper) - dist.cdf(xi_lower)

            #i starts 1 until p (length of subsequence)
            for i in range(1, len(subsequence)):
                if delta != 0:
                    xi_lower, xi_upper = subsequence[i] - delta, subsequence[i] + delta
                    ximinus1_lower, ximinus1_upper = subsequence[i-1] - delta, subsequence[i-1] + delta

                #P(A|B) = P(A ^ B)/P(B)
                #P(5|3) = p(3^5)/P(3)
                if self.model == "empirical":
                    if delta == 0:
                        count = np.sum((time_series[:-1] == subsequence[i-1]) & (time_series[1:] == subsequence[i]))
                        logging.debug(count)
                    else:
                        count = np.sum(
                            (np.logical_and(time_series[:-1] >= ximinus1_lower, time_series[:-1] <= ximinus1_upper)) &
                            (np.logical_and(time_series[1:] >= xi_lower, time_series[1:] <= xi_upper))
                        )
                        logging.debug(count)
                    numerator = count / (len(time_series) - 1)
                    if delta == 0:
                        count = np.sum(time_series == subsequence[i-1])
                    else:
                        count = np.sum(np.logical_and(time_series >= ximinus1_lower, time_series <= ximinus1_upper))
                    denominator = count / len(time_series)
                elif self.model == "kde":
                    numerator = dist_bivar.integrate_box([ximinus1_lower, xi_lower], [ximinus1_upper, xi_upper])
                    denominator = dist.integrate_box_1d(ximinus1_lower, ximinus1_upper) * dist.integrate_box_1d(xi_lower, xi_upper)

                elif self.model == "gaussian_theoretical":
                    numerator = dist_bivar.cdf([ximinus1_upper, xi_upper]) - dist_bivar.cdf([ximinus1_lower, xi_lower])
                    denominator = dist.cdf(ximinus1_upper) - dist.cdf(ximinus1_lower)

                p_Q_j *= min(1, numerator / denominator)

            logger.debug(subsequence)
            logger.debug("p_Q_j = %E", p_Q_j)
            p_Q *= p_Q_j

        return p_Q
    
    #TODO: Assuming variables are independent and first-order markov between time points
    def vars_dep_time_markov(self, motif_subsequence, variables):
        return 0
    
    #TODO:Assuming variables are independent and independence between time points
    def var_indep_time_indep():
        return 
    
    @staticmethod
    def hochberg_critical_value(p_values, false_discovery_rate=0.05):
        ordered_pvalue = sorted(p_values)

        critical_values = []
        for i in range(1,len(ordered_pvalue)+1):
            critical_values.append((i/len(ordered_pvalue)) * false_discovery_rate)

        critical_val = ordered_pvalue[len(ordered_pvalue)-1]
        for i in reversed(range(len(ordered_pvalue))):
            if ordered_pvalue[i] < critical_values[i]:
                critical_val = ordered_pvalue[i]
                break
        if critical_val > 0.05:
            critical_val = 0.05
        return critical_val
    
    @staticmethod
    def bonferonni_correction(p_values, alpha=0.05):
        return alpha/len(p_values)
    
class Motif:
    multivar_sequence = []
    variables = []
    delta_thresholds = []
    n_matches = 0
    p_Q = 0
    pvalue = 1

    def __init__(self, multivar_sequence, variables, delta_thresholds, n_matches, pattern_probability=0, pvalue=1):
        self.multivar_sequence = multivar_sequence
        self.variables = variables
        self.delta_thresholds = delta_thresholds
        self.n_matches = n_matches


    def set_pattern_probability(self, model, vars_indep=True):
        p = 0
        if vars_indep:
            p = model.vars_indep_time_markov(self.multivar_sequence, self.variables, self.delta_thresholds)
        else:
            p = model.vars_dep_time_markov(self.multivar_sequence, self.variables, self.delta_thresholds)

        self.p_Q = p
        return p
    
    def set_significance(self, max_possible_matches, data_n_variables, idd_correction=False):
    
        if self.p_Q == 0:
            return 0
        if self.p_Q == 1:
            return 1
        pvalue = 0
        if self.n_matches < max_possible_matches:
            for j in range(self.n_matches, max_possible_matches+1):
                try:
                    pvalue += binom.pmf(j, max_possible_matches, self.p_Q)
                except OverflowError:
                    pvalue += 0      
        else:
            return np.nan
        
        if idd_correction:
            pvalue = min(1, pvalue * math.comb(data_n_variables, len(self.variables)))

        self.pvalue = pvalue
        logging.info("p_value = %.3E (p_pattern = %.3E)", self.pvalue, self.p_Q)
        return pvalue
    


    
