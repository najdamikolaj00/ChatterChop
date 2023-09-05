"""This file contains functions that can be used to perform audio transcriptions and additional utilities."""

import os
import whisper

from .utils import (
    text_normalization,
    create_edit_matrix,
    levenshtein_distance
)

def whisper_transcription(path_to_audio_file: str, model_name: str = 'small', language: str = 'pl') -> dict:
    """
    Given a path to an audio file, perform transcription and return transcription result. 
    OpenAI Whisper Copyright (c) 2022 OpenAI https://github.com/openai/whisper on MIT LICENSE

    Args:
        path_to_audio_file (str): A path to the audio file.
        model_name (str, optional): The name of the Whisper model. Default is 'small'.
        language (str, optional): The language shortcut based on the language of the audio file. Default is 'pl'.

    Returns:
        dict: whisper transcription dictionary that includes transcription text and detected language.
    """
    try:
        model = whisper.load_model(model_name)
    except FileNotFoundError:
        return {"error": f"Model '{model_name}' not found"}

    try:
        transcription_result = model.transcribe(path_to_audio_file, language=language)
        return transcription_result
    except Exception as e:
        return {"error": f"Transcription failed: {str(e)}"}

def transcription_into_txt(txt_path: str, transcription_result: dict):
    """
    Function for writing transcription into a text file.
    If the text file or directory doesn't exist, it will be created.

    Args:
        txt_path (str): Path to the text file or directory where it will be created.
        transcription_result (dict): Transcription result in a dictionary.

    Outcome:
        Text file with a transcription.
    """
    try:
        if os.path.isdir(txt_path):
            txt_path = os.path.join(txt_path, "transcription.txt")

        os.makedirs(os.path.dirname(txt_path), exist_ok=True)

        with open(txt_path, "a", encoding="utf-8") as file_txt:
            file_txt.write(f"{transcription_result['text']}\n")
    except Exception as e:
        print(f"Error writing to the text file: {str(e)}")

def transcription_into_list(transcription_result: dict) -> list:
    """
    If transcription isn't stored in the dictionary then this function writes transcript to the 
    list as one string. Needed when the metrics are calculated.

    Args:
        transcription_result (dict): Transcription result dictionary.

    Returns:
        list: The whole transcription in a one element's list.

    *Note: Before calculating any metric it is necessary to perform normalization of capitalization and 
    punctuation from a transcription.
    """
    try:
        text = transcription_result['text'].read()
        one_long_sentence = text_normalization(text)
        sentence_list = [one_long_sentence]
        return sentence_list
    
    except FileNotFoundError:
        print(f"Transcription not found: returning empty list.")
        return []
    
def txt_into_list(path_to_text: str) -> list:
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
        with open(path_to_text, 'r', encoding='utf-8') as file:
            text = file.read()
            one_long_sentence = text_normalization(text)
            sentence_list = [one_long_sentence]
            return sentence_list
    except FileNotFoundError:
        print(f"File not found: {path_to_text}")
        return []

def wer_metric(transcription_result: list, ground_truth: list) -> float:
    """
    Function calculates Word Error Rate (WER) https://en.wikipedia.org/wiki/Word_error_rate.
    
    Args:
        transcription_result (list): The one element's list with a transcription result.
        ground_truth (list): The one element's list with a ground truth.

    Returns:
        float: Word Error Rate (WER).

    """
    ref_tokens = transcription_result.split()
    hyp_tokens = ground_truth.split()


    edit_matrix = create_edit_matrix(ref_tokens, hyp_tokens)
    
    levenshtein_matrix = levenshtein_distance(ref_tokens, hyp_tokens, edit_matrix)

    wer = levenshtein_matrix[len(ref_tokens)][len(hyp_tokens)] / len(ref_tokens)

    return wer

def cer_metric(transcription_result: list, ground_truth: list) -> float:
    """
    Function calculates Character Error Rate (CER) https://readcoop.eu/glossary/character-error-rate-cer/.
    
    Args:
        transcription_result (list): The one element's list with a transcription result.
        ground_truth (list): The one element's list with a ground truth.

    Returns:
        float: Character Error Rate (CER).

    """
    ref_chars = list(ground_truth)
    hyp_chars = list(transcription_result)

    edit_matrix = create_edit_matrix(ref_chars, hyp_chars)
    
    levenshtein_matrix = levenshtein_distance(ref_chars, hyp_chars, edit_matrix)

    cer = levenshtein_matrix[len(ref_chars)][len(hyp_chars)] / len(ref_chars)

    return cer