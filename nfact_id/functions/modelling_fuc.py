import numpy as np
import xarray as xr
import pandas as pd 
import arviz as az

def calculate_effect_size(mean, y_sigma):
    return mean/y_sigma

def define_rope_width(y_sigma):
    return  0.1 * y_sigma.mean().item() 

def calculate_rope(y_sigma, posterior):
    rope_width = define_rope_width(y_sigma)
    param = posterior.values.flatten()

    inside_rope = np.mean((param > -rope_width) & (param < rope_width))
    return {
        "inside_rope": inside_rope,
        "outside_rope": 1 - inside_rope
    }
def diagnostics(posterior, posterior_to_check):
    rope = calculate_rope(posterior['value_sigma'], posterior[posterior_to_check])
    print(rope['inside_rope']*100, "Inside ROPE")
    print(rope['outside_rope']*100, "Outside of ROPE")
    print(calculate_effect_size( posterior[posterior_to_check].mean().item(), posterior['value_sigma'].mean().item()), " effect size")


def get_zscores_distrubtions(idata_object: xr.Dataset, subject_component_vals: np.ndarray) -> xr.Dataset:
    """
    Function to get the zscore distrubtion
    from a posterior predictive.

    Parameters
    ----------
    idata_object: xr.Dataset
        fitted model with posterior
        predictive
    subject_component_vals: np.ndarray
        array of subject values
    
    Returns
    -------
    xr.Dataset
    """
    ppd = idata_object.posterior_predictive["value"]
    new_sub_values = xr.DataArray(
        subject_component_vals,
        dims=("value_obs",),
        coords={"value_obs": ppd.coords["value_obs"]}
        )
    val = (new_sub_values - ppd) / ppd.std(dim=("chain", "draw"))
    return val.rename("zscores")

def get_zscore_summary(z_scores: xr.Dataset) -> pd.DataFrame:
    """
    Function to get a summary of zscores
    from zscores distrubtion

    Parameters
    ----------
    z_scores: xr.Dataset

    Returns
    -------
    pd.DataFrame: Dataframe
        DataFrame of mean, hdi_2.5%
        and hdi_97.5%
    """
    z_mean = z_scores.mean(dim=("chain","draw")).values
    z_hdi = az.hdi(z_scores, hdi_prob=0.95)
    return pd.DataFrame({
        "mean": z_mean,
        'hdi_2.5%': z_hdi.sel(hdi="lower")["x"].values,
        "hdi_97.5%": z_hdi.sel(hdi="higher")["x"].values
    })

def rope_mass(posterior: xr.DataArray, rope:tuple =(-0.1, 0.1)) -> dict:
    """
    A rope mass diagnostic function. 
    Returns ROPE mass, posterior odds
    (basically a bayes factor) and bayes factor 
    for the null hypothesis (that there 
    is no effect, i.e everything is inside ROPE)

    Parameters
    -----------
    posterior: xr.DataArray
        posterior data
    rope: tuple
        tuple of rope range (-0.1, 0.1)
    
    Returns
    -------
    dict: dictionary
       dict of inside, above and below
       rope mass, posterior odds and
       null bayes factor
    """
    lower, upper = rope
    inside = np.mean((posterior > lower) & (posterior < upper))   # fraction 0–1
    below  = np.mean(posterior <= lower)
    above  = np.mean(posterior >= upper)
    outside = below + above
    posterior_odds = outside / inside if inside > 0 else np.inf
    bf_01 = 1 / posterior_odds 
    return {
        "inside": inside*100, 
        "below": below*100, 
        "above": above*100, 
        "posterior_odds": posterior_odds, 
        "bf_01": bf_01
    }

def sum_across_chains(idata: xr.DataArray):
    """
    Function to sum across chains

    Parameters
    -----------
    idata: xr.Dataset
        data of the chains
    
    Returns
    -------
    xr.DataArray: DataArray
        Dataarry of summed chains
    """
    return idata.mean(dim=("chain"))

def single_submation_statistic(idata: xr.DataArray, dim_1: int, dim_2: int) -> xr.DataArray:
    """
    Function to sum across the first dim.
    So give it number of subjects 
    as first dim 1 to sum across 
    subjects or components to sum across
    components.

    Parameters
    ---------
    idata: xr.Dataset
        data of the chains
    dim_1: int
        number of values in
        the first dimension. 
        This is the dimension to 
        sum across
    dim_2: int
        number of values in
        the 2nd dimension. 
    
    Returns
    -------
    xr.DataArray: DataArray
        Dataarry of summed across 
        1st dim
    """
    zscores= sum_across_chains(idata)
    z_scores_reshaped = zscores.data.reshape(dim_2, dim_1, zscores.draw.size)
    return xr.DataArray(
        z_scores_reshaped,
        dims=( "dim2", "dim1", "draw")
    ).mean(dim="dim1")