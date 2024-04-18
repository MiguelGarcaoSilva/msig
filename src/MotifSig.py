import numpy as np
import math
import decimal as dc
import sys
from scipy.stats import norm, binom, gaussian_kde, multivariate_normal
import logging

logger = logging.getLogger(__name__)

class NullModel:
    data = None
    pre_computed_distribution = {}
    pre_computed_bivariate_distribution = {}
    model = None

    def __init__(self, data, model="empirical_dist"):
        self.data = data
        self.model = model

        for var_index, y_j in enumerate(data):
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

            elif self.model == "empirical":
                pass
            else:
                raise ValueError("Invalid model")


    def vars_indep_time_markov(self, motif_subsequence, variables, delta_thresholds):
        p = 1
        # for variable in motif:
        for j, subsequence in enumerate(motif_subsequence):   
            var_index = variables[j]
            delta = delta_thresholds[var_index]
            p_Q = 1
            time_series = self.data[var_index]
            if self.model != "empirical":
                dist = self.pre_computed_distribution[var_index]
                dist_bivar = self.pre_computed_bivariate_distribution[var_index]

            for i in range(0, len(subsequence)):
                xi_lower, xi_upper = subsequence[i] - delta, subsequence[i] + delta
                ximinus1_lower, ximinus1_upper = subsequence[i-1] - delta, subsequence[i-1] + delta

                if i == 0:
                    if self.model == "empirical":
                        if delta == 0:
                            count = np.sum(time_series == subsequence[i])
                        else:
                            count = np.sum(np.logical_and(time_series >= xi_lower, time_series <= xi_upper))
                        p_Q *= count / len(time_series)
                    elif self.model == "kde":
                        p_Q *= dist.cdf(xi_upper) - dist.cdf(xi_lower)
                    elif self.model == "gaussian_theoretical":
                        p_Q *= dist.cdf(xi_upper) - dist.cdf(xi_lower)
                else:
                    #P(A|B) = P(A ^ B)/P(B)
                    #P(5|3) = p(3^5)/P(3)
                    if self.model == "empirical":
                        if delta == 0:
                            count = np.sum((time_series[:-1] == subsequence[i-1]) & (time_series[1:] == subsequence[i]))
                        else:
                            count = np.sum(
                                (np.logical_and(time_series[:-1] >= ximinus1_lower, time_series[:-1] <= ximinus1_upper)) &
                                (np.logical_and(time_series[1:] >= xi_lower, time_series[1:] <= xi_upper))
                            )
                        numerator = count / (len(time_series) - 1)
                        if delta == 0:
                            count = np.sum(time_series == subsequence[i-1])
                        else:
                            count = np.sum(np.logical_and(time_series >= ximinus1_lower, time_series <= ximinus1_upper))
                        denominator = count / len(time_series)
                    elif self.model == "kde":
                        numerator = dist_bivar.cdf([ximinus1_upper, xi_upper]) - dist_bivar.cdf([ximinus1_lower, xi_lower])
                        denominator = dist.cdf(ximinus1_upper) - dist.cdf(ximinus1_lower)

                    elif self.model == "gaussian_theoretical":
                        numerator = dist_bivar.cdf([ximinus1_upper, xi_upper]) - dist_bivar.cdf([ximinus1_lower, xi_lower])
                        denominator = dist.cdf(ximinus1_upper) - dist.cdf(ximinus1_lower)

                    p_Q *= min(1, numerator / denominator)

            logger.debug("p_Q = %E", p_Q)
            p *= p_Q

        return p
    
    #TODO: implement
    def vars_dep_time_markov(self, motif_subsequence, variables):
        return 0
    
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
    match_indices = []
    n_matches = 0
    motif_probability = 0
    pvalue = 1

    def __init__(self, multivar_sequence, variables, delta_thresholds, match_indices, pattern_probability=0, pvalue=1):
        self.multivar_sequence = multivar_sequence
        self.variables = variables
        self.delta_thresholds = delta_thresholds
        self.match_indices = match_indices
        self.n_matches = len(match_indices)

    @staticmethod 
    def bin_prob(n, p, k):
        if p==1: return 1
        ctx = dc.Context()
        arr = math.factorial(n) // math.factorial(k) // math.factorial(n-k)
        bp = (dc.Decimal(arr) * ctx.power(dc.Decimal(p), dc.Decimal(k)) * ctx.power(dc.Decimal(1-p), dc.Decimal(n-k)))
        return float(bp) if sys.float_info.min < bp else sys.float_info.min
    

    def set_pattern_probability(self, model, vars_indep=True):
        p = 0
        if vars_indep:
            p = model.vars_indep_time_markov(self.multivar_sequence, self.variables, self.delta_thresholds)
        else:
            p = model.vars_dep_time_markov(self.multivar_sequence, self.variables, self.delta_thresholds)

        self.motif_probability = p
        return p
    
    def set_significance(self, max_possible_matches, data_n_variables, idd_correction=False):
        pvalue = 0
        if self.n_matches < max_possible_matches:
            #pvalue = sum(self.bin_prob(max_possible_matches, self.motif_probability, j) for j in range(self.n_matches, max_possible_matches+1))  
            pvalue = sum(binom.pmf(j, max_possible_matches, self.motif_probability) for j in range(self.n_matches, max_possible_matches+1))  
        else:
            return np.nan
        
        if idd_correction:
            pvalue = min(1, pvalue * math.comb(data_n_variables, len(self.variables)))

        self.pvalue = pvalue
        logging.info("p_value = %E (p_pattern = %E)", self.pvalue, self.motif_probability)
        return pvalue
    


    