"""This file contains functions additional utilities for a package."""

from typing import List
import re
import torch
import torchaudio

def func_template(value: int, text: str, values: list) -> str:
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

def load_audio_tc(audio_path: str) -> tuple(torch.Tensor, int):
    """
    Loading audio when working with torch.

    Args:
        audio_path (str): Path to audio file.

    Returns:
        tuple(torch.Tensor, int): Waveform and sample rate of audio file.

    """

    waveform, sample_rate = torchaudio.load(audio_path)
    return waveform, sample_rate

def waveform_resample_tc(waveform: torch.Tensor, sample_rate: int, desired_sample_rate: int = 16000) -> torch.Tensor:
    """
    Waveform resample function when working with torch.

    Args:
        waveform (torch.Tensor): Waveform as torch tensor.
        sample_rate (int): Original sample rate of waveform.
        desired_sample_rate (int): Desired sample rate for waveform.

    Returns:
        torch.Tensor: Resampled waveform to the desired sample rate.

    """
    transform = torchaudio.transforms.Resample(orig_freq = sample_rate,
                                                new_freq= desired_sample_rate)
    resampled_waveform = transform(waveform)

    return resampled_waveform

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
    normalized_transcript = re.sub(r'(\w*)[^\w\s]+(\w*)', r'\1 \2', transcript)
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


