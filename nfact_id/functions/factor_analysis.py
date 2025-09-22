from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity
from factor_analyzer import ConfirmatoryFactorAnalyzer, ModelSpecification
from sklearn.metrics import root_mean_squared_error, mean_squared_error
import pandas as pd
import numpy as np
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def obtain_correlation_values(df, corr_val) -> pd.Series:
    """
    Function to obtain all values
    greater than or equal to
    a given correlation.

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe of values
    corr_val: float
        correlation value to use as
        threshold

    Returns
    -------
    filtered_corr: pd.Series
       Series of correlation
       values greater than
       or equal to threshold
    """
    filtered_corr = (
        df.corr().apply(lambda x: x.where((x >= corr_val) & (x < 1))).stack()
    )
    return filtered_corr[
        filtered_corr.index.get_level_values(0)
        < filtered_corr.index.get_level_values(1)
    ]


def matrix_determinate(df: pd.DataFrame) -> int:
    """
    Function to check determinate of matrix

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe of values

    Returns
    -------
    int: integer
        int of determinate
    """
    return np.linalg.det(np.corrcoef(df, rowvar=False))


def check_variables(df: pd.DataFrame, factor_names: list) -> None:
    """
    Function to do a basic ANOVA with post hoc testing
    to check that

    Parameters
    ----------
    df: pd.DataFrame
        dataframe of values
    factor_names: list
        name of factors

    Returns
    -------
    None
    """
    for variable in factor_names:
        groups = [
            df[variable][df["phenotype"] == group] for group in df["phenotype"].unique()
        ]
        f_stat, p_value = f_oneway(*groups)
        print(f"ANOVA for {variable}: F={f_stat:.2f}, p={p_value:.4f}")

        # If p < 0.05, perform post hoc test
        if p_value < 0.05:
            tukey = pairwise_tukeyhsd(
                endog=df[variable], groups=df["phenotype"], alpha=0.05
            )
            print("\n", tukey)


def cfa_model(
    loadings: np.ndarray,
    normalized_data: np.ndarray,
    n_factors: int,
    n_variables: int,
    factor_names: list,
) -> object:
    """
    Function to create and fit
    a cfa model.

    Parameters
    ----------
    loadings: np.ndarray
        loadings design matrix
    normalized_data: np.ndarray
        scaled data to fit model to
    n_factors: int
        number of factors
    n_variables: int
        number of variables
    factor_names: list
        list of factor names

    Returns
    -------
    model: object
        fitted ConfirmatoryFactorAnalyzer
        model
    """
    cfa_model = ModelSpecification(
        loadings=loadings,
        n_factors=n_factors,
        n_variables=n_variables,
        factor_names=factor_names,
    )
    model = ConfirmatoryFactorAnalyzer(cfa_model, disp=False)
    return model.fit(normalized_data)


def kmo_scoring(data: np.ndarray, cols: list) -> dict:
    """
    Function to calculate the kmo
    score for data for factor
    analysis (higher i.e >0.7 )

    Parameters
    ----------
    data: np.ndarray
        array of data to
        be used in FA
    cols: list
        list of columns

    Returns
    -------
    dict: dictionary object
        dict of kmo values,
        kmo_all & kmo_model
    """
    kmo_all, kmo_model = calculate_kmo(data)
    return {
        "kmo_all": pd.DataFrame(kmo_all, index=cols, columns=["KMO Score"]),
        "kmo_model": kmo_model,
    }


def data_fit(df) -> dict:
    """
    Function to test if the data
    is apprioriate for factor
    analysis

    Parameters
    ----------
    df: pd.DataFrame
        data for factor
        analysis

    Returns
    -------
    dict: dictionary object
        dict of kmo values,
        kmo_all & kmo_model and
        bartlett_pval & bartlett_chi2
    """

    chi2, pval = calculate_bartlett_sphericity(df.values)
    kmo_model = kmo_scoring(df.values, df.columns)
    return {
        "kmo_all": kmo_model["kmo_all"],
        "kmo_model": kmo_model["kmo_model"],
        "bartlett_pval": pval,
        "bartlett_chi2": chi2,
    }


def create_loading_design_matrix(loading_dict: dict, raw_df_columns: list):
    """
    Function to create a loadings
    desing matrix for the CFA.

    Given dict ={
    "FA1": [col1, col3]
    "FA2": [col2,  col4]
    }

    This function will return
    array([
       [1., 0.],
       [0., 1.],
       [1., 0.],
       [1., 0.],
       ])

    Parameters
    ----------
    loading_dict: dict
        loading dictionary
    raw_df_columns: list
        list of raw data
        columns

    """
    loadings = np.zeros((len(raw_df_columns), len(loading_dict)))
    for factor_idx, (_, variables) in enumerate(loading_dict.items()):
        for var in variables:
            var_idx = list(raw_df_columns).index(var)
            loadings[var_idx, factor_idx] = 1
    return loadings


def model_fit_parameters(model: object, data: np.ndarray, cols: list) -> dict:
    """
    Function to calculate
    basic model fit parameters
    and loadings

    Parameters
    ----------
    model: object
        fitted ConfirmatoryFactorAnalyze
        instance
    data: np.ndaraay:
        array of data that has been fitted
    cols: list
        list of column names

    Returns
    -------
    dict of loadings, rmse and
    mse
    """
    true_cov = np.cov(data.T)
    predicted_cov = model.get_model_implied_cov()
    return {
        "loadings": pd.DataFrame(model.loadings_.T, columns=cols),
        "rmse": root_mean_squared_error(true_cov, predicted_cov),
        "mse": mean_squared_error(true_cov, predicted_cov),
    }
