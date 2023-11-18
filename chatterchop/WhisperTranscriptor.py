from typing import Optional

from chatterchop.Transcription import Transcription


class WhisperTranscriptor:
    """
    Given a path to an audio file, perform transcription and return transcription result.
    OpenAI Whisper Copyright (c) 2022 OpenAI https://github.com/openai/whisper on MIT LICENSE

    Args:
        model_name (str, optional): The name of the Whisper model. Default is 'small'.
        language (str, optional): The language shortcut based on the language of the audio file. Default is 'pl'.

    Returns:
        str: whisper transcription text
    """

    def __init__(self, language: Optional[str] = None, model_name: str = "small"):
        self.language = language
        self.model_name = model_name
        try:
            self.model = whisper.load_model(model_name)
        except FileNotFoundError:
            raise FileNotFoundError(f"Model '{model_name}' not found")

    def transcript(self, path_to_audio_file: str) -> Transcription:
        """
        Given a path to an audio file, perform transcription and return transcription result.
        OpenAI Whisper Copyright (c) 2022 OpenAI https://github.com/openai/whisper on MIT LICENSE

        Args:
            path_to_audio_file (str): A path to the audio file.

        Returns:
            Transcription (str): whisper transcription text
        """

        try:
            transcription_result = self.model.transcribe(
                path_to_audio_file, language=self.language
            )
            return Transcription((transcription_result["text"]))
        except Exception as e:
            raise Exception(f"Transcription failed: {str(e)}")
