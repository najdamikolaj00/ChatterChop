"""Core ChatterChop class and all the corresponding methods."""

import torchaudio
import os

from .transcription import WhisperTranscription

from .forced_alignment import (
    run_forced_alignment
)

class ChatterChop(WhisperTranscription):
    """
    The core class of the package provides the necessary methods for correct usage. 
    Within a class object initialisation, the user is able to call methods for 
    transcription and forced alignment.

    """
    def __init__(self, path_to_audio, transcript=None):
        """
        Initialize the ChatterChop object.

        Args:
            path_to_audio (str): Path to the audio file to load.
            transcript (str, optional): Path to a transcript file (default is None, then 
                                        whisper transcription is performed).
        """
        self._waveform = None
        self._sample_rate = None
        self._desired_sample_rate = 16000
        self.path_to_audio = path_to_audio
        
        self._load_audio(self.path_to_audio)

        if transcript is None:
            self.transcript = self.whisper_transcription()
            print(type(self.transcript))
        else:
            self.transcript = transcript

        if self.transcript is not None and self._waveform is not None:
            self._segments, self._trellis_size_0 = run_forced_alignment(self._waveform, self.transcript)
        else:
            print("Transcript and waveform must be specified")

        self._chopped_chatter = []

    @property
    def waveform(self):
        """Property to get the waveform."""
        return self._waveform

    @property
    def sample_rate(self):
        """Property to get the sample rate."""
        return self._sample_rate

    def _load_audio(self, path):
        """
        Load audio from a file and optionally resample it if necessary.
        The preferred sampling rate is 16kHz due to the sampling rate of audio
        data on which the model has been trained.

        Args:
            path (str): Path to the audio file to load.
        """
        try:
            self._waveform, self._sample_rate = torchaudio.load(path)
            
            if self._sample_rate != self._desired_sample_rate:
                self._waveform = self._resample_waveform(self._desired_sample_rate)
                self._sample_rate = self._desired_sample_rate

        except FileNotFoundError:
            print(f"File not found: {path}")
            self._waveform, self._sample_rate = None, None
        except Exception as e:
            print(f"An error occurred while loading audio: {e}")
            self._waveform, self._sample_rate = None, None

    def _resample_waveform(self, desired_sample_rate):
        """
        Resample waveform to 16kHz, this is the sample rate preferred for forced 
        alignment as the model was trained on audio samples with that sampling rate.

        Args:
            desired_sample_rate (int): Desired sample rate for waveform.

        Returns:
            torch.Tensor: Resampled waveform.
        """
        try:
            transform = torchaudio.transforms.Resample(orig_freq=self._sample_rate,
                                                    new_freq=desired_sample_rate)
            return transform(self._waveform)
        except Exception as e:
            print(f"An error occurred during resampling: {e}")

    def get_transcription(self):
        """Returns transcription."""
        return self.transcript

    def chop_chatter(self):
        """
        Takes a speech file and cuts it into chunks 
        by time frames obtained by forced alignment.

        """
        for i in range(len(self._segments)):
            ratio = self._waveform.size(1)/(self._trellis_size_0 - 1)
            word = self._segments[i]
            x0 = int(ratio * word.start)
            x1 = int(ratio * word.end)

            entry = {
                'word': word.label,
                'audio_segment': self._waveform[:, x0:x1]
            }

            self._chopped_chatter.append(entry)
        
    def save_speech_segments(self, output_path):
        """
        Saves chopped audio in desired path using 
        recognised/transcripted word in uppercase as a filename.

        Args:
            output_path (str): Path to the specific output directory.

        """
        for segment in self._chopped_chatter:
            torchaudio.save(os.path.join(output_path, f'{segment["word"]}.wav'), 
                            segment['audio_segment'], sample_rate=self._desired_sample_rate)