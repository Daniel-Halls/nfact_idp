from nfact_id.functions.behavioural_functions import nn
import pandas as pd
import numpy as np


def task_transform_and_merge(
    input_df: pd.DataFrame, input_column: str, participant_info: pd.DataFrame
):
    """
    Function to transform and merge HCP tasks
    of the HCP pipeline. Assumes
    HCP style naming

    Parameters
    ------------
    input_df: pd.DataFrame
         Input dataframe
    input_column: str
         The column to transform
    participant_info: pd.DataFrame
        A seperate dataframe DataFrame
        containing participant information

    Returns
    --------
    variable_df_score: pd.DataFrame
        Transformed and merged DataFrame
    """
    variable_df = input_df[["src_subject_id", input_column]].drop_duplicates(
        subset=["src_subject_id"]
    )
    variable_df = pd.merge(
        variable_df, participant_info, left_on="src_subject_id", right_on="id"
    )

    transformed_data = {}
    groups = variable_df.groupby("phenotype")

    for group_id, group in groups:
        transformed = nn(group[[input_column]])
        dataframe = pd.DataFrame(np.round(transformed), columns=[input_column])
        transformed_data[group_id] = pd.concat(
            [
                group[["src_subject_id", "phenotype"]].reset_index(drop=True),
                dataframe.reset_index(drop=True),
            ],
            axis=1,
        )

    variable_df_score = pd.concat(
        [transformed_data[group] for group in transformed_data]
    ).sort_values("src_subject_id")

    return variable_df_score.rename(columns={"src_subject_id": "id"}).sort_values(
        by="id"
    )


def process_questionnaire(
    input_df: pd.DataFrame,
    particpant_info: pd.DataFrame,
    id_column: str,
    time_point: str = "T1",
) -> pd.DataFrame:
    """
    Function to process questionnaire

    Parameters
    ----------
    input_df: pd.DataFrame
        input dataframe
    particpant_info: pd.DataFrame
        particpant_info dataframe
    id_column: str
       id_colum to remove
    time_point: str
        Which time point. Default is T1

    Returns
    --------
    variable_df: pd.DataFrame
        processed dataframe
    """
    try:
        variable_df = input_df[input_df["visit"] == time_point].drop_duplicates(
            subset=["src_subject_id"]
        )
    except Exception:
        variable_df = input_df.drop_duplicates(subset=["src_subject_id"])
    return pd.merge(
        variable_df.drop(
            ["collection_id", id_column, "dataset_id", "subjectkey"], axis=1
        ),
        particpant_info,
        left_on="src_subject_id",
        right_on="id",
    ).drop(["visit", "respondent", "collection_title", "Study"], axis=1)


def questionnaire_transform_merge(
    input_df: pd.DataFrame,
    participant_info: pd.DataFrame,
    id_column: str,
    score_name: str,
    column_list: list,
):
    """
    Function to transform and merge HCP tasks
    of the HCP pipeline. Assumes
    HCP style naming

    Parameters
    -----------
    input_df: pd.DataFrame
         Input dataframe
    participant_info: pd.DataFrame
        A seperate dataframe DataFrame
        containing participant information
    id_column: str
        The column representing unique subject IDs
    score_name: str:
        The name of the transformed score column
    column_list: list
        List of columns to use for transformation

    Returns
    --------
    variable_df_score: pd.DataFrame
        Transformed and merged DataFrame
    """
    try:
        variable_df = input_df[input_df["visit"] == "T1"].drop_duplicates(
            subset=["src_subject_id"]
        )
    except Exception:
        variable_df = input_df.drop_duplicates(subset=["src_subject_id"])
    variable_df = pd.merge(
        variable_df.drop(
            ["collection_id", id_column, "dataset_id", "subjectkey"], axis=1
        ),
        participant_info,
        left_on="src_subject_id",
        right_on="id",
    ).drop(["visit", "respondent", "collection_title", "Study", "id"], axis=1)

    transformed_data = {}
    groups = variable_df.groupby("phenotype")

    for group_id, group in groups:
        transformed = nn(group[column_list])
        dataframe = pd.DataFrame(np.round(transformed)).sum(axis=1)

        transformed_data[group_id] = pd.concat(
            [
                group[["src_subject_id", "phenotype"]].reset_index(drop=True),
                dataframe.reset_index(drop=True),
            ],
            axis=1,
        )

    variable_df_score = pd.concat(
        [transformed_data[group] for group in transformed_data]
    ).sort_values("src_subject_id")

    return variable_df_score.rename(
        columns={"src_subject_id": "id", 0: score_name}
    ).sort_values(by="id")


def pdc_get_timepoint_1(
    df: pd.DataFrame, particpant_info: pd.DataFrame
) -> pd.DataFrame:
    """
    Function to get first time point
    of pdc data.

    Parameters
    ----------
    df: pd.DataFrame
        dataframe of pdc
    particpant_info: pd.DataFrame
        pdc neuroimaging subjects

    Returns
    ------
    pd.Dataframe: dataframe
        dataframe of t1 particpants
        of which there is nueorimging data
    """
    subject_counts = df["src_subject_id"].value_counts()
    duplicate_subjects = subject_counts[subject_counts > 1].index
    earliest_indices = (
        df[df["src_subject_id"].isin(duplicate_subjects)]
        .groupby("src_subject_id")["interview_date"]
        .idxmin()
    )
    unique_subjects = subject_counts[subject_counts == 1].index
    non_duplicate_rows = df[df["src_subject_id"].isin(unique_subjects)]
    df_filtered = pd.concat([df.loc[earliest_indices], non_duplicate_rows])
    df_filtered = df_filtered.sort_values(by="src_subject_id").reset_index(drop=True)
    return pd.merge(
        df_filtered,
        particpant_info[["id", "phenotype"]],
        left_on="src_subject_id",
        right_on="id",
        how="right",
    )
