from scipy.stats import beta
import xarray as xr
import arviz as az
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

def beta_distribution(outside_total: int, total_number: int) -> xr.DataArray:
    """
    Function to get the beta 
    distribtuion given number of samples
    outside a set condition 
    (i.e number of subjects outside ROPE) and total number
    of samples (i.e total number of population)

    Parameters
    ----------
    outside_total: int
        int of the number of samples
        outside a given condition
    total_number: int
        total number of samples
    
    Returns
    -------
    xr.DataArray: DataArray
        xr data array of posterior
        distrubtion
    """
    alpha =  1 + outside_total
    beta_param = 1 + total_number - outside_total
    posterior_samples = beta.rvs(alpha, beta_param, size=4000)
    return az.from_dict(posterior={"theta": posterior_samples})

def mask_condition(z_scores: pd.DataFrame) -> pd.Series:
    """
    Function to create a mask 
    based on if the mean and hdi are outside 
    of ROPE (strictest)

    Parameter
    ---------
    z_scores: pd.DataFrame
        z_scores
    
    Returns
    -------
    pd.Series: Series
        Series of bool values
    """
    return (
        ((z_scores['hdi_2.5%'] > 0.1) & (z_scores['hdi_97.5%'] > 0.1))
        | ((z_scores['hdi_2.5%'] < -0.1) & (z_scores['hdi_97.5%'] < -0.1))
    ) & (z_scores['mean'].abs() > 0.1)

def perc_affected_from_cohen_d(cohen_d: np.ndarray) -> np.ndarray:
    """
    Function to get a 
    non-overlap percentage given a 
    cohen_ds distrubtion

    Parameters
    ----------
    cohen_d: np.ndarray
        distrubtion of posterior
        given a set cohens d value
    
    Returns
    -------
    np.ndarray: array
        distrubtion of percentage value
    """
    overlap = 2 * norm.cdf(-np.abs(cohen_d)/2)
    return (1 - overlap) * 100

def get_posterior_for_d(
        cohen_d: float, 
        se:float = 0.05, 
        prior_mean: float = 0.0, 
        prior_sd: float = 0.5, 
        n_samps: int= 2000) -> np.ndarray:
    """
    Function to get posterior given 
    a set cohens d

    Parameters
    ----------
    cohen_d: float
        effect size 
    se: float
        standard error
        Default is 0.05
    prior_mean: float = 0.0
        mean of prior 
        distribution
    prior_sd: float = 0.5
        standard deviation
        of prior
    n_samps: int 
        number samples to draw 
        from the posterior
        Default is 2000.

    Returns
    --------
    np.ndarray: array
        Posterior distribution
    """
    prior_var = prior_sd**2
    lik_var = se**2
    post_var = 1/(1/prior_var + 1/lik_var)
    post_mean = post_var * (prior_mean/prior_var + cohen_d/lik_var)
    post_sd = np.sqrt(post_var)
    return np.random.normal(post_mean, post_sd, n_samps)


def get_expected_value(
        cohen_d: float, 
        se:float = 0.05, 
        prior_mean: float = 0.0, 
        prior_sd: float = 0.5, 
        n_samps: int= 2000) -> np.ndarray:
    """
    Function to get posterior given 
    a set cohens d

    Parameters
    ----------
    cohen_d: float
        effect size 
    se: float
        standard error
        Default is 0.05
    prior_mean: float = 0.0
        mean of prior 
        distribution
    prior_sd: float = 0.5
        standard deviation
        of prior
    n_samps: int 
        number samples to draw 
        from the posterior
        Default is 2000.

    Returns
    --------
    np.ndarray: array
        Posterior distribution
    """
    dist = get_posterior_for_d(
        cohen_d, 
        se, 
        prior_mean, 
        prior_sd, 
        n_samps)
    percentage_distribution = perc_affected_from_cohen_d(dist)
    return {
        "mean": np.mean(percentage_distribution),
        "hdi_95": np.percentile(percentage_distribution, [2.5, 97.5])
    }