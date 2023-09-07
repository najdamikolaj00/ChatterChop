"""This file contains functions that can be used to perform audio transcriptions and additional utilities."""
import whisper

from .utils import (
    wer_metric,
    cer_metric
)

class WhisperTranscription:
    def __init__(self, path_to_audio_file):

        self.path_to_audio = path_to_audio_file
        self.transcription_result = None

    def whisper_transcription(self, model_name='small', language='pl'):
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
            return self.transcription_result['text']
        except Exception as e:
            return {"error": f"Transcription failed: {str(e)}"}
        
    def get_transcription(self):
        """Returns transcription."""
        return self.transcription_result['text']
    
    def get_transcription_accuracy(self, ground_truth):
        """
        Returns transcription accuracy based on WER and CER metric.
            Args:
                ground_truth (list): The one element's list with a ground truth.

            Returns:
                print(WER and CER results in %)
        """
        if self.transcription_result is not None:
            wer = wer_metric(self.get_transcription, ground_truth)
            cer = cer_metric(self.get_transcription, ground_truth)

            print(f"WER: {wer*100:.2f}%, CER:{cer*100:.2f}%")
        else:
            print("None")