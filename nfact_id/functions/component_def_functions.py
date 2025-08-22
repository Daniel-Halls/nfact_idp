from collections import defaultdict
import re 
import pandas as pd

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