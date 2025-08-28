import pandas as pd

def create_correlation_matrix(df: pd.DataFrame, component_type: str) -> pd.DataFrame:
    """
    Function to create correlation matrix 
    out of component loadings.

    Parameters
    ----------
    df: pd.DataFrame
        dataframe of loadings
    component_type: str
        component type

    Returns
    -------
    pd.DataFrame: corr matrix
        correlation matrix of sub by sub
    """
    return df[[col for col in df.columns if component_type in col]].T.corr()

def create_wide_df(grey_path: str, white_path: str) -> pd.DataFrame:
    """
    Function to create a wide dataframe
    of component loadings for G and W

    Parameters
    ----------
    grey_path: str
        path to G comp loadings 
    white_path: str
        path to W comp loadings

    Returns
    -------
    pd.DataFrame: DataFrame
        wide dataframe of 
        comp loadings

    """
    g_comp = pd.read_csv(grey_path)
    w_comp = pd.read_csv(white_path)
    cols_to_rename = [col for col in w_comp.columns if 'comp' in col]
    g_comp.rename(columns={col: "G_" + col for col in cols_to_rename}, inplace=True)
    w_comp.rename(columns={col: "W_" + col for col in cols_to_rename}, inplace=True)
    return pd.concat([g_comp, w_comp.iloc[:, 1:]], axis=1).reset_index(drop=True)
