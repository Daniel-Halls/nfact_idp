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
    yi = xr.DataArray(
            subject_component_vals,
            dims=("value_obs",),
            coords={"value_obs": ppd.coords["value_obs"]}
            )

    expected_variation = ppd.std(dim=("chain", "draw"))  
    residuals = yi - ppd
    sigma_n_j = residuals.std(dim="value_obs") 
    val = residuals / np.sqrt(expected_variation**2 + sigma_n_j**2)
    return val.rename("zscores")

def get_zscores_distrubtions_old_way(idata_object: xr.Dataset, subject_component_vals: np.ndarray) -> xr.Dataset:
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

def check_values(val: int) -> bool:
    """
    Function to check a given value
    is > 0

    Parameters
    -----------
    val: int
        interger to check
    
    Returns
    -------
    bool: Boolean
        True > 0 else
        False
    """
    return True if val > 0 else False

def get_posterior_odds(alt_val: int, null_val: int) -> int:
    """
    Function to get posterior 
    odds.

    Parameters
    ----------
    alt_val: int
        alternative value
        interger
    null_val: int
        null value
    
    Returns
    -------
    int: interger
        posterior odds. 
    """
    if not check_values(alt_val): return -9999
    if not check_values(null_val): return 9999
    return alt_val / null_val 

def get_log_posterior_odds(alt_val: int, null_val: int) -> int:
    """
    Function to get log posterior 
    odds.

    Parameters
    ----------
    alt_val: int
        alternative value
        interger
    null_val: int
        null value
    
    Returns
    -------
    int: interger
        log posterior
        odds. 
    """
    if not check_values(alt_val): return 9999
    if not check_values(null_val): return -9999
    return np.log(null_val / alt_val)

def get_rope_mass(posterior: xr.DataArray, rope: tuple) -> dict:
    """
    Function to get rope mass

    Parameters
    ----------
    posterior: xr.DataArray
        distrubtion to 
        check
    rope: tuple
        rope width
    
    Returns
    -------
    dict: dictionary object
        dict of inside and outside

    """
    lower, upper = rope
    inside = np.mean((posterior > lower) & (posterior < upper))   
    below  = np.mean(posterior <= lower)
    above  = np.mean(posterior >= upper)
    outside = below + above
    return {
        "inside": inside,
        "outside": outside
    }

def rope_mass(posterior: xr.DataArray, rope: tuple =(-0.1, 0.1)) -> dict:
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
    posterior = posterior.to_numpy()
    rope_mass = get_rope_mass(posterior, rope)
 
    posterior_odds = get_posterior_odds(
        rope_mass['outside'], 
        rope_mass['inside'])

    log_odds_null = get_log_posterior_odds(rope_mass['outside'], rope_mass['inside']) 
    mean = posterior.mean()
    return {
        "mean": mean,
        "inside": rope_mass['inside']*100, 
        "outside": rope_mass['outside']*100,
        "posterior_odds": posterior_odds, 
        "bf_01": 1 / posterior_odds,
        "log_po": log_odds_null
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

def reshape_posteriors(zscores: xr.Dataset, no_subjects: int, no_components: int) -> xr.DataArray:
    """
    Function to reshape posteriors into
    'wide' format 

    Paramaters
    ----------
    zscores: xr.DataArray
        xr.Dataset with zscores in
        them
    no_subjects: int
        number of subjects
    no_components: int
        number of components

    Returns
    --------
    xr.DataArray: dataarray
        data array of
        subject, component
        chain, draw
    """

    return xr.DataArray(
        zscores.data.reshape(
            no_subjects, 
            no_components, 
            zscores.chain.size, 
            zscores.draw.size),
        dims=( "sub", "comp", "chain", "draw")
    )

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
    return az.summary(z_scores, hdi_prob=0.95, kind='stats')[['mean', 'hdi_2.5%', "hdi_97.5%"]].reset_index(drop=True)

def get_meaningful_values(z_posteriors: xr.DataArray, df_comp_vals: pd.DataFrame, log_po: int = -2.94, meaningful: int = 0.1, rope=(-0.1, 0.1)) -> list:
    """
    Function to see if individual
    z score posteriors are meaningful

    Parameters
    ----------
    z_posteriors: xr.DataArray
        DataArray of posteriors 
        to loop over
    log_po: int = -2.94
        log_po cut off 
    meaningful: int = 0.1
        mean cut off
    
    Returns
    -------
    list: list 
        list of indcies
        of meaningful values

    """
    df = df_comp_vals.copy(deep=True)
    df['z_score'] = None
    for idx, sub_comp in enumerate(z_posteriors.stack(samples=("chain", "draw"))):
        mass = rope_mass(sub_comp, rope=rope)
        if np.abs(mass['mean']) < meaningful:
            df.loc[idx, 'z_score'] = 0
            continue
        if mass['log_po'] > log_po:
            df.loc[idx, 'z_score'] = 0
            continue
        df.loc[idx, 'z_score'] = df.loc[idx, 'mean']
    return df

def is_meaningful(z_posteriors: xr.DataArray, log_po: int = -2.94, meaningful: int = 0.1, rope=(-0.1, 0.1 )) -> list:
    """
    Function to see if individual
    z score posteriors are meaningful

    Parameters
    ----------
    z_posteriors: xr.DataArray
        DataArray of posteriors 
        to loop over
    log_po: int = -2.94
        log_po cut off 
    meaningful: int = 0.1
        mean cut off
    
    Returns
    -------
    list: list 
        list of indcies
        of meaningful values

    """
    return [
    sub for sub in range(z_posteriors.shape[0])
    if (
        (abs((mass_dict := rope_mass(z_posteriors[sub].stack(sample=("chain", "draw"))))["mean"]) > meaningful)
        and (mass_dict["log_po"] <= log_po)
    )]

def subject_meaningul_components(means_df: pd.DataFrame, posterior: xr.DataArray, controls_data: pd.DataFrame, no_comp: int) -> pd.DataFrame:
    """
    Function to get meaningful subject
    distrubtions given the a set of controls

    Parameters
    ---------- 
    means_df: pd.DataFrame
        dataframe with mean zscores 
        from the posterior sample
    posterior: xr.DataArray
        the posterior sample
    controls_data: pd.DataFrame
        Dataframe of controls
        hdi 
    no_comp: int
        number of components
    
    Returns
    -------
    component_df: pd.DataFrame
        data frame of mean value
        if the posterior sample
        is meaningul else 0
    """
    component_df = means_df[['subject', 'group']].copy(deep=True)
    for comp in range(no_comp):
        comp_posterior = posterior[:, comp]
        comp_is_meaninfgul = get_meaningful_values(comp_posterior, means_df[["subject", comp]].rename(columns={comp: 'mean'}), 
                                               rope=(controls_data['hdi_2.5%'][comp], controls_data['hdi_97.5%'][comp]))
        component_df[comp] = comp_is_meaninfgul['z_score'].values
    return component_df
