"""Core ChatterChop class and all the corresponding methods."""

import torchaudio
import os

from .transcription import (
    whisper_transcription
)

from .forced_alignment import (
    get_word_segments
)

class ChatterChop:
    """
    Core class of the package, the methods for necessary initialization of audio signal and resampling
    are provided. 
    With class initialization user is able to call methods for transcription
    and forced alignment. 

    """
    def __init__(self, path_to_audio):
        """
        Initialize the ChatterChop object.

        Args:
            path (str): Path to the audio file to load.
        """
        self._waveform = None
        self._sample_rate = None
        self._desired_sample_rate = 16000
        
        self.load_audio(path_to_audio)

        self._transcript = whisper_transcription(path_to_audio)
        self._segments, self._trellis_size_0 = get_word_segments(self._waveform, self._transcript)
        self._chopped_chatter = []

    @property
    def waveform(self):
        """Property to get the waveform."""
        return self._waveform

    @property
    def sample_rate(self):
        """Property to get the sample rate."""
        return self._sample_rate

    def load_audio(self, path):
        """
        Load audio from a file, and optionally resample it if needed.

        Args:
            path (str): Path to the audio file to load.
        """
        try:
            self._waveform, self._sample_rate = torchaudio.load(path)
            
            if self._sample_rate != self._desired_sample_rate:
                self._waveform = self.resample_waveform(self._desired_sample_rate)
                self._sample_rate = self._desired_sample_rate

        except FileNotFoundError:
            print(f"File not found: {path}")
            self._waveform, self._sample_rate = None, None
        except Exception as e:
            print(f"An error occurred while loading audio: {e}")
            self._waveform, self._sample_rate = None, None

    def resample_waveform(self, desired_sample_rate):
        """
        Resample waveform to 16kHz, this is the sample rate needed for forced alignment
        as the model was trained on audio samples with that frequency sampling.

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

    def chop_chatter(self):
        """
        take speech file and cut it into chunks by segment time frames
        save it into dict?
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
        save chopped audio using word as a name of a file.
        """
        for segment in self._chopped_chatter:
            torchaudio.save(os.path.join(output_path, f'{segment["word"]}.wav'), 
                            segment['audio_segment'], sample_rate=self._desired_sample_rate)