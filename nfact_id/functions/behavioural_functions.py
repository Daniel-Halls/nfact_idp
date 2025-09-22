import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
import numpy as np
from functools import reduce
import warnings
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")


def clustering(n_clusters: int, data: pd.DataFrame) -> object:
    """
    Function to return a fitted k means
    cluster model.

    Parameters
    ----------
    n_clusters: int
        number of clusters
    data: pd.DataFrame
        data to perform clustering on

    Returns
    -------
    kmeans: KMeans object
        fitted kmeans object
    """
    kmeans = KMeans(init="k-means++", n_clusters=n_clusters, n_init=4)
    kmeans.fit(data)
    return kmeans


def fit_pca(no_comp: int, normalised_data: pd.DataFrame) -> np.ndarray:
    """
    Function to perform PCA

    Parameters
    ----------
    no_comp: int
       no of components
    normalised_data: pd.DataFrame
        normalised dataframe

    Returns
    -------
    np.ndarray: array
        array of PCA values
    """

    return PCA(n_components=no_comp).fit_transform(normalised_data)


def determine_number_of_clusters(
    pca_data: np.ndarray, cluster_algo: object, cluster_range: tuple = range(2, 11)
) -> None:
    """
    Function to calculate
    the silhouette score
    for clusters to determine the
    number of clusters that should be used

    Parameters
    ----------
    pca_data: np.ndarray
        data to perform clustering on
    cluster_algo: object
        Which cluster algo to use
    cluster_range: tuple
        Default is range(2, 11)
    """
    for cluster_n in cluster_range:
        clusterer = cluster_algo(n_clusters=cluster_n, random_state=10)
        cluster_labels = clusterer.fit_predict(pca_data)
        silhouette_avg = silhouette_score(pca_data, cluster_labels)
        print(
            "For n_clusters =",
            cluster_n,
            "The average silhouette_score is :",
            silhouette_avg,
        )


def pca_permulation(
    pca_df: pd.DataFrame, normalised_data: pd.DataFrame, n_perms: int = 10000
) -> int:
    """
    Function to perform a PCA permuation
    approach

    Parameters
    ----------
    pca_df: pd.DataFrame
        dataframe to perform PCA on
    normalised_data: pd.DataFrame
        dataframe of normalised data
    n_perms: int=10000
        number of permuations

    Returns
    -------
    int: integer
        the number of signficiant
        PCA components
    """

    decomp = PCA()
    decomp.fit_transform(normalised_data)
    null_distro = permutation_null_distro(pca_df, n_perms=n_perms)
    crti_val = get_crit_val(len(decomp.explained_variance_ratio_), null_distro)
    alt_val = get_explained_ratio(decomp, len(decomp.explained_variance_ratio_))
    return len(get_significant_components(crti_val, alt_val))


def impute_group_median(group: pd.DataFrame):
    """
    Function to impute the median value
    by group.

    Parameters
    --------
    group: pd.DataFrame
        dataframe of group values

    Returns
    -------
    group: pd.DataFrame
        dataframe with median imputed values
    """
    for col in group.select_dtypes(include=["object", "number"]).columns:
        if col not in ["id", "phenotype"]:
            group[col] = pd.to_numeric(group[col], errors="coerce")
            group[col] = group[col].fillna(group[col].median())
    return group


def merge_dataframes(df_list: list):
    """
    Function to merge a list
    of dataframes into a
    single large dataframe.

    Parameters
    ----------
    df_list: list
        list of dataframes

    Returns
    -------
    pd.DataFrame: dataframe
        merged dataframe
    """
    return reduce(
        lambda left, right: pd.merge(left, right, on="id", how="outer"), df_list
    )


def convert_df_type(df: pd.DataFrame):
    """
    Function to convert dataframe type
    into float.

    Parameters
    ----------
    df: pd.Dataframe
        dataframe to convert

    Returns
    -------
    df: pd.Dataframe
        converted dataframe
    """
    for col in df.columns:
        try:
            assert df[col].dtypes != "object"
        except AssertionError:
            try:
                df[col] = df[col].astype("float")
            except ValueError:
                continue
    return df


def scaled_data(df) -> np.ndarray:
    """
    Function wrapper around scklearn
    StandardScaler.

    Parameters
    ----------
    df: pd.DataFrame
        dataframe to scale

    Returns
    -------
    np.ndarray: np.array
        array of scaled values
    """
    return StandardScaler().fit_transform(df.values)


def scaling(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to return Z scored
    data for PCA

    Parameters
    ----------
    df: np.array
        Matrix of values

    Returns
    -------
    pd.DataFrame:
        DataFrame of scaled values
    """

    cols = []
    for col in df.columns:
        if df[col].dtype == "float":
            cols.append(col)

    scaled_df_data = scaled_data(df[cols])
    scaled_df_data = pd.DataFrame(
        scaled_df_data,
        columns=cols,
    )
    scaled_df_data["sex"] = df["sex"].reset_index(drop=True)
    return scaled_df_data


def PCA_analysis(data: np.array) -> object:
    """
    Function to do PCA

    Parameters
    ----------
    data: np.array
        array of data to do PCA on

    Returns
    -------
    decomp: object
       PCA model

    """
    return PCA().fit(data)


def permutation_null_distro(data: pd.DataFrame, n_perms: int = 5000) -> np.array:
    """
    Function to permute the null distribution

    Parameters
    ----------
    data: pd.DataFrame
        data to permuate

    n_perms: int=5000
        number of permuations

    Returns
    -------
    explained_variance_perm: np.array
        array of null distribution for each
        component
    """
    explained_variance_perm = np.zeros((n_perms, data.shape[1]))
    for perm in range(n_perms):
        perm_data = data.copy()
        for col in range(data.shape[1]):
            perm_data.iloc[:, col] = np.random.permutation(perm_data.iloc[:, col])
        perm_data = scaling(perm_data)
        pca_perm = PCA_analysis(perm_data)
        explained_variance_perm[perm] = pca_perm.explained_variance_ratio_
    return explained_variance_perm


def get_crit_val(number_of_components: int, null_distro: np.array) -> dict:
    """
    Function to determine crit val

    Parameters
    ----------
    number_of_components: int
        number of components to check
    null_distro: np.array
        array of the null distibution

    Returns
    -------
    crti_val: dict
        dictionary of criticial values
    """
    crit_val = {}
    for comp in range(number_of_components):
        null_distribution = null_distro[:, comp]
        if max(null_distribution) > 0 and min(null_distribution) < 0:
            crit_val[comp] = np.abs(np.quantile(null_distribution, 0.975))
        if min(null_distribution) > 0:
            crit_val[comp] = np.quantile(null_distribution, 0.95)
        if max(null_distribution) <= 0:
            crit_val[comp] = np.quantile(null_distribution, 0.05)
    return crit_val


def get_explained_ratio(alt_pca: object, n_comp: int):
    """
    Function to organise the explained ratio for
    comparison

    Parameters
    ----------
    alt_pca: object
        fitted sklearn.decomposition.PCA object
    n_comp: int
        number of components
    """
    ratio = {}
    for comp in range(n_comp):
        ratio[comp] = alt_pca.explained_variance_ratio_[comp]
    return ratio


def get_significant_components(crti_val: dict, alt_val: dict) -> list:
    """
    Funciton to get significant components

    Parameters
    ----------
    crti_val: dict
        dictionary of criticial values
    alt_val: dict

    Returns
    -------

    """
    components = []
    for comp in crti_val.keys():
        if alt_val[comp] > crti_val[comp]:
            components.append(comp)
    return components


def imputer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to impute dataframe by
    KMNIImputer

    Parameters
    ---------
    df: pd.DataFrame
        DataFrame

    Returns
    -------
    pd.DataFrame: imputed DataFrame
    """
    imputed_data = {}
    groups = df.groupby("phenotype")
    for group in groups.all().index:
        group_df = groups.get_group(group)
        imputed = KNNImputer().fit_transform(group_df[group_df.columns[4:]].values)
        data = pd.concat(
            [
                group_df[
                    ["src_subject_id", "phenotype", "interview_age", "sex"]
                ].reset_index(drop=True),
                pd.DataFrame(imputed).reset_index(drop=True),
            ],
            axis=1,
        )
        data.columns = group_df.columns
        imputed_data[group] = data
    return pd.concat(imputed_data.values())


def nn(score) -> np.array:
    """
    Function wrapper around the
    KNNImputer function in
    sckit learn.

    Parameters
    ----------
    score: np.array
        array of values

    Returns
    -------
    np.array: array
        array of imputed
        scores
    """
    return KNNImputer(missing_values=np.nan, weights="uniform").fit_transform(score)
