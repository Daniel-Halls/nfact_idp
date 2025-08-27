from collections import defaultdict
import re 
import pandas as pd
import numpy as np
from conilab.stats.stats import calculate_dice


def clean_and_group(inner_dict: dict) -> dict:
    """
    Function to group and clean the atlas json
    files for saving to 
    
    Parameters
    ----------
    inner_dict: dict
        dictionary to be cleaned

    Parameters
    ----------
    grouped: dict
        grouped dictionary
    """
    grouped = defaultdict(float)
    for key, value in inner_dict.items():
        key = key.strip()
        match = re.match(r'^(.*?)(?:\s?[A-Z]|\s?\d+)?$', key)
        if match:
            base_key = match.group(1)
            grouped[base_key] = (grouped[base_key] + value) / 2 if base_key in grouped else value
    return dict(grouped)

def get_max_value_from_dict(dict_val: dict) -> dict:
    """
    Function to get max value from 
    a nested dictionary.

    Parameters
    -----------
    dict_val: dictionary
        Nested dictionary
        of values

    Returns
    -------
    dict: dictionary 
        Unested dict of 
        max value of nested 
        dict 

    """
    return {
    outer_key: (
        max(inner_dict.items(), key=lambda item: item[1])[0]
        if inner_dict and max(inner_dict.values()) > 0.0
            else "Unknown"
    )
    for outer_key, inner_dict in dict_val.items()
    }

def most_frequent(row: pd.Series) -> str:
    """
    Function to get the most common
    str in a dataframe. Returns
    no_consensus

    Parameters
    ----------
    row: pd.Series
        pd series of a row
    
    Returns
    -------
    str: string
        str of most common string 
        object 
    """
    counts = row.value_counts()
    if counts.max() == 1: 
        return "no_consensus"
    else:
        return counts.idxmax()
    

def rename_dict(dict_to_clean: dict, key_rename_map: str) -> dict:

    """
    Function to rename dictionary key

    Parameters
    -----------
    dict_to_clean: dict
        dictionary to clean
    key_rename_map: str
        what to rename key to

    Returns
    -------
    dict_to_clean: dict 
        dictionary to clean
    """
    for outer_key, inner_dict in dict_to_clean.items():
        updated = {}
        for k, v in inner_dict.items():
            new_key = key_rename_map.get(k, k)  # Rename if in map, otherwise keep same
            updated[new_key] = v
        dict_to_clean[outer_key] = updated
    
    return dict_to_clean

def get_network_value(labels: pd.DataFrame, label_name: str) -> list:
    """
    Function to index for a network given a set name

    Parameters
    -----------
    labels: pd.DataFrame
        dataframe with two
        columns named
        Network Name and 
        Network Order
    label_name: str
        network name

    Returns
    --------
    list: list
       list of indices
    """
    return labels[labels['Network Name'].str.contains(label_name)]['Network Order'].values.tolist()

def get_network_indices(label_name: str, labels: pd.DataFrame, atlas: np.ndarray) -> dict:
    """
    Function to get indices of a network indices

    Parameters
    ----------
    label_name: str
        string of label name
    labels: pd.DataFrame
        dataframe of label name 
        with two columns Network Name
        and Network Order
    left_atlas: np.ndarray
        array of values representing
        networks
    right_atlas: np.ndarray
        array of values representing
        networks

    Returns
    -------
    dict: dictionary
        dictionary of indices
        associated with a network
        divided into left and right
    """
    label_val = get_network_value(labels, label_name)
    return np.where(np.isin(atlas, label_val))

def define_components_by_dice(networks: list, g_nmf_data: dict, component_range: int, labels: pd.DataFrame, atlas: np.ndarray, subcortical: bool=True) -> dict:
    """
    Function to define components by getting the median value of
    the nmf by network

    Parameters
    ----------
    networks: list
        list of networks
    g_nmf_data: dict
        dictionary of grey matter data
    component_range: int
        number of components
    label_name: str
        string of label name
    labels: pd.DataFrame
        dataframe of label name 
        with two columns Network Name
        and Network Order
    atlas: np.ndarray
        array of values representing
        networks

    Returns
    -------
    dict: dictionary
        dict of component and median 
        value of network
    """
    comp_dict = dict(zip([comp for comp in range(component_range)], [{} for _ in range(component_range)]))
    for comp in range(component_range):
        l_surf = g_nmf_data['L_surf'][:,comp]
        r_surf = g_nmf_data['R_surf'][:,comp]
        nmf_distriubtion = np.concatenate([l_surf, r_surf])
        if subcortical:
            subcortcal_nmf = g_nmf_data['vol'].get_fdata()[:,:,:, comp].flatten()
            nmf_distriubtion = np.concatenate([nmf_distriubtion, subcortcal_nmf])
        
        for network in networks:
            index = get_network_indices(network, labels, atlas)
            atlas_mask = np.zeros_like(nmf_distriubtion, dtype=int)
            atlas_mask[index] = 1
            nmf_mask = np.zeros_like(nmf_distriubtion, dtype=int)
            nmf_index = np.where(nmf_distriubtion >0)
            nmf_mask[nmf_index] = 1
            score = calculate_dice(nmf_mask, atlas_mask)
            comp_dict[comp][network] = float(score)
    return comp_dict