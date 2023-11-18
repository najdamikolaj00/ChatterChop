"""Core ChatterChop class and all the corresponding methods."""
import os
from pathlib import Path

from chatterchop.AudioLoader import AudioLoader
from chatterchop.ChoppedChatter import ChoppedChatter
from chatterchop.WhisperTranscriptor import WhisperTranscriptor
from chatterchop.forced_alignment import run_forced_alignment


class ChatterChopper:
    """
    The core class of the package provides the necessary methods for correct usage.
    Within a class object initialisation, the user is able to call methods for
    transcription and forced alignment.

    """

    def __init__(
        self,
        audio_loader: AudioLoader = None,
        whisper_transcriptor: WhisperTranscriptor = None,
    ):
        """
        Initialize the ChatterChop object.

        Args:
            path_to_audio (str): Path to the audio file to load.
            transcript (str, optional): Path to a transcript file (default is None, then
                                        whisper transcription is performed).
        """
        if whisper_transcriptor is None:
            whisper_transcriptor = WhisperTranscriptor()
        if audio_loader is None:
            audio_loader = AudioLoader()
        self.waveform = None
        self.transcript = None
        self._audio_loader = audio_loader
        self._whisper_transcriptor = whisper_transcriptor

    def chop_chatter(
        self, path_to_audio: str = None, transcript: str | Path = None
    ) -> ChoppedChatter:
        """
        Takes a speech file and cuts it into chunks
        using time frames obtained by forced alignment.

        """
        self.waveform = self.waveform or self._audio_loader.load_audio(path_to_audio)
        if os.path.isfile(transcript):
            self.transcript = transcript.read_text()
        self.transcript = self.transcript or self._whisper_transcriptor.transcript(
            path_to_audio
        )
        chopped_chatter = ChoppedChatter(
            self._audio_loader.desired_sample_rate, self.transcript
        )
        _segments, _trellis_size_0 = run_forced_alignment(
            self.waveform, self.transcript
        )

        for i in range(len(_segments)):
            ratio = self.waveform.size(1) / (_trellis_size_0 - 1)
            word = _segments[i]
            x0 = int(ratio * word.start)
            x1 = int(ratio * word.end)

            entry = {"word": word.label, "audio_segment": self.waveform[:, x0:x1]}

            chopped_chatter.append(entry)
        return chopped_chatter
