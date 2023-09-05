"""This file contains functions additional utilities for a package."""

from typing import List

def func_tempalte(value: int, text: str, values: list) -> str:
    """
    Function description

    Args:
        value (int): Description.
        text (str): Description.
        values (list): Description.

    Returns:
        str: Description.

    """
    return 'a'

def text_normalization(text: str) -> str:
    """
    This function performs text normalization techniques, 
    to provide a clear representation of text for metrics 
    such as WER or CER.

    Args:
        text (str): Text to normalize.

    Returns:
        str: Normalized text as a one long sentence.

    *To do: change numbers to text
        
    """
    text = text.strip()
    text = ''.join(char.lower() if char.isalnum() or char.isspace() else ' ' for char in text)
    one_long_sentence = ' '.join(text.split())

    return one_long_sentence

def min_of_three(a: any, b: any, c: any) -> any:
    """
    Helper function for WER and CER metrics. 
    The minimum value of three given values.

    Args:
        a (any): First value to compare.
        b (any): Second value to compare.
        c (any): Third value to compare.

    Returns:
        any: Description.

    """
    return min(a, min(b, c))

def create_edit_matrix(reference: List[str], hypothesis: List[str]) -> List[List[int]]:
    """
    This function creates a matrix that is used to 
    calculate Levenshtein distance.

    Args:
        reference (List[str]): The reference text (ground truth).
        hypothesis (List[str]): The transcription text. 

    Returns:
        List[List[int]]: Edit matrix.

    """
    edit_matrix = [[0] * (len(hypothesis) + 1) for _ in range(len(reference) + 1)]

    for i in range(len(reference) + 1):
        edit_matrix[i][0] = i
    for j in range(len(hypothesis) + 1):
        edit_matrix[0][j] = j

    return edit_matrix

def levenshtein_distance(reference: List[str], hypothesis: List[str], matrix: List[List[int]]) -> List[List[int]]:
    """
    Function to calculate Levenshtein distance.

    Args:
        reference (List[str]): The reference text (ground truth).
        hypothesis (List[str]): The transcription text. 
        matrix (List[List[int]]): Edit matrix to perform Levenshtein distance calculation.

    Returns:
       List[List[int]]: Calculated Levenshtein distance.

    """
    for i in range(1, len(reference) + 1):
        for j in range(1, len(hypothesis) + 1):
            cost = 0 if reference[i - 1] == hypothesis[j - 1] else 1
            matrix[i][j] = min_of_three(
                matrix[i - 1][j] + 1,
                matrix[i][j - 1] + 1,
                matrix[i - 1][j - 1] + cost
            )
    return matrix


