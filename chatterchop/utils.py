"""This file contains functions additional utilities for a package."""

from typing import List
import re
import torch

def check_cuda_availability():
    """
    Checking if cuda is available.

    Args:
        None

    Returns:
        device

    """
    if torch.cuda.is_available():
        print("CUDA is available.")
        device = torch.device("cuda")
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        print("CUDA is not available. Using CPU.")
        device = torch.device("cpu")

    return device

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

def normalize_polish_letters(transcript):
    polish_to_english = str.maketrans({
    'ą': 'a',
    'ć': 'c',
    'ę': 'e',
    'ł': 'l',
    'ń': 'n',
    'ó': 'o',
    'ś': 's',
    'ź': 'z',
    'ż': 'z',
    })
    transcript = transcript.lower()

    eng_transcript = transcript.translate(polish_to_english)

    return eng_transcript

def normalize_transcript_CTC(transcript: str) -> str:
    """
    This function performs text normalization for forced alignment.
    Example: 
        transcript: "I had that curiosity beside me at thit moment."
        normalized: "I|HAD|THAT|CURIOSITY|BESIDE|ME|AT|THIS|MOMENT"

    Args:
        transcript (str): Transcript to normalize.

    Returns:
        str: Normalized transcript prepared for forced alignment.

    *To do: change numbers to text
        
    """

    eng_transcript = normalize_polish_letters(transcript)
    normalized_transcript = re.sub(r'(\w*)[^\w\s]+(\w*)', r'\1 \2', eng_transcript)
    normalized_transcript = '|'.join(word.upper() for word in normalized_transcript.split())

    return normalized_transcript

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


