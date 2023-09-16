"""This file contains functions used to perform audio transcriptions and additional utilities."""
import whisper

from .utils import (
    wer_metric,
    cer_metric,
    transcription_into_txt
)

class WhisperTranscription:
    def __init__(self, path_to_audio_file):

        self.path_to_audio = path_to_audio_file
        self.transcription_result = self.transcript

    def whisper_transcription(self, language, model_name='small'):
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
            self.transcription_result = model.transcribe(self.path_to_audio, language=language)
            self.transcription_result = self.transcription_result['text']
            return self.transcription_result
        except Exception as e:
            return {"error": f"Transcription failed: {str(e)}"}
            
    def transcription_accuracy(self, transcription, ground_truth):
        """
        Returns transcription accuracy based on WER and CER metric.
            Args:
                ground_truth (str): Path to file containing ground truth.

            Prints:
                WER and CER results in % or None if didn't meet the requirements. 
        """
        if transcription is not None:
            wer = wer_metric(transcription, ground_truth)
            cer = cer_metric(transcription, ground_truth)

            print(f"WER: {wer*100:.2f}%, CER:{cer*100:.2f}%")
        else:
            print("None")

    def save_transcription(self, txt_path):
        """Function that writes transcription into a text file."""
        return transcription_into_txt(self.transcription_result, txt_path)