"""This file contains functions additional utilities for a package."""

import re
import torch
import os

polish_to_english = str.maketrans({'ą': 'a', 'ć': 'c', 'ę': 'e', 
                                       'ł': 'l', 'ń': 'n', 'ó': 'o', 
                                       'ś': 's','ź': 'z', 'ż': 'z',})

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

def text_normalization(text):
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

def normalize_polish_to_english(transcript):
    """
    This function performs the translation of letters from Polish to English 
    using the provided mapper (polish_to_english) as a reference.

    Args:
        transcript (str): Transcript containing Polish letters.

    Returns:
        str: Normalised transcript that contains specified English equivalent 
        letters in Polish.

    *To do: change numbers to text
        
    """

    transcript = transcript.lower()
    return transcript.translate(polish_to_english)

def normalize_transcript_CTC(transcript):
    """
    This function performs text normalisation for forced alignment.

    Example: 
        transcript: "Nieznaną łąką podąża załamany człowiek."
        normalized: "NIEZNANA|LAKA|PODAZA|ZALAMANY|CZLOWIEK"

    Args:
        transcript (str): Transcript to normalize.

    Returns:
        str: Normalized transcript prepared for forced alignment.

    *To do: change numbers to text, other languages
        
    """
    has_polish_letters = any(polish_letter in transcript for polish_letter in polish_to_english)

    if has_polish_letters:
        eng_transcript = normalize_polish_to_english(transcript)
    
    
    normalized_transcript = re.sub(r'(\w*)[^\w\s]+(\w*)', r'\1 \2', eng_transcript)
    normalized_transcript = '|'.join(word.upper() for word in normalized_transcript.split())

    return normalized_transcript

def transcription_into_txt(transcript, txt_path):
    """
    Function for writing transcription into a text file.
    If the text file or directory doesn't exist, it will be created.

    Args:
        txt_path (str): Path to the text file or directory where it will be created.

    Outcome:
        Text file with a transcription.
    """
    try:
        if os.path.isdir(txt_path):
            txt_path = os.path.join(txt_path, "transcription.txt")

        os.makedirs(os.path.dirname(txt_path), exist_ok=True)

        with open(txt_path, "a", encoding="utf-8") as file_txt:
            file_txt.write(f"{transcript}\n")
    except Exception as e:
        print(f"Error writing to the text file: {str(e)}")

def transcription_into_list(transcript):
    """
    If transcription isn't stored in a list then this function writes transcript to the 
    list as one string. Needed when the metrics are calculated.

    Returns:
        list: The whole transcription in a one element's list.

    *Note: Before calculating any metric it is necessary to perform normalization of capitalization and 
    punctuation from a transcription.
    """
    try:
        one_long_sentence = text_normalization(transcript)
        sentence_list = [one_long_sentence]
        return sentence_list
    
    except FileNotFoundError:
        print(f"Transcription not found: returning empty list.")
        return []

def txt_into_list(path_to_text):
    """
    This function converts text file into list and performs preparation for metrics calculation.

    Args:
        path_to_text (str): A path to a text file.
    Returns:
        list: The whole text in a one element's list.

    *Note: Before calculating any metric it is necessary to perform normalization of capitalization,
    punctuation and numbers conversion from a ground truth.
    """
    try:
        text = reading_a_txt_file(path_to_text)
        one_long_sentence = text_normalization(text)
        sentence_list = [one_long_sentence]
        return sentence_list
    except FileNotFoundError:
        print(f"File not found: {path_to_text}")
        return []
    
def txt_into_str(path_to_text):
    """
    This function returns string contained in a file.

    Args:
        path_to_text (str): A path to a text file.
    Returns:
        str: The whole text in one long sentence.

    *Note: Before calculating any metric it is necessary to perform normalization of capitalization,
    punctuation and numbers conversion from a ground truth.
    """
    try:
        with open(path_to_text, 'r', encoding='utf-8') as file:
            text = file.read()
            one_long_sentence = text_normalization(text)
            return one_long_sentence
    except FileNotFoundError:
        print(f"File not found: {path_to_text}")
        return ""
    
def wer_metric(transcription_result, ground_truth):
    """
    Function calculates Word Error Rate (WER) https://en.wikipedia.org/wiki/Word_error_rate.
    
    Args:
        transcription_result (str): Transcription result.
        ground_truth (str): Path to a file containing the ground truth text.

    Returns:
        float: Word Error Rate (WER).

    """
    ref_tokens = transcription_result.split()
    hyp_tokens = ground_truth.split()
 
    edit_matrix = create_edit_matrix(ref_tokens, hyp_tokens)
    
    levenshtein_matrix = levenshtein_distance(ref_tokens, hyp_tokens, edit_matrix)

    wer = levenshtein_matrix[len(ref_tokens)][len(hyp_tokens)] / len(ref_tokens)

    return wer

def cer_metric(transcription_result, ground_truth):
    """
    Function calculates Character Error Rate (CER) https://readcoop.eu/glossary/character-error-rate-cer/.
    
    Args:
        transcription_result (str): Transcription result.
        ground_truth (str): Path to a file containing the ground truth text.

    Returns:
        float: Character Error Rate (CER).

    """

    if ground_truth.endswith('.txt'):
        ground_truth = txt_into_str(ground_truth)

    ref_chars = list(ground_truth)
    hyp_chars = list(transcription_result)

    edit_matrix = create_edit_matrix(ref_chars, hyp_chars)
    
    levenshtein_matrix = levenshtein_distance(ref_chars, hyp_chars, edit_matrix)

    cer = levenshtein_matrix[len(ref_chars)][len(hyp_chars)] / len(ref_chars)

    return cer
    
def min_of_three(a, b, c):
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

def create_edit_matrix(reference, hypothesis):
    """
    This function creates a matrix that is used to 
    calculate Levenshtein distance.

    Args:
        reference (list): The reference text (ground truth).
        hypothesis (list): The transcription text. 

    Returns:
        list: Edit matrix.

    """
    edit_matrix = [[0] * (len(hypothesis) + 1) for _ in range(len(reference) + 1)]

    for i in range(len(reference) + 1):
        edit_matrix[i][0] = i
    for j in range(len(hypothesis) + 1):
        edit_matrix[0][j] = j

    return edit_matrix

def levenshtein_distance(reference, hypothesis, matrix):
    """
    Function to calculate Levenshtein distance.

    Args:
        reference (list): The reference text (ground truth).
        hypothesis (list): The transcription text. 
        matrix (list): Edit matrix to perform Levenshtein distance calculation.

    Returns:
       list: Calculated Levenshtein distance.

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

def is_file_path(path):
    """Checks if the path contains a file."""
    return os.path.isfile(path)

def reading_a_txt_file(path):
    """Returns a text."""
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text

