import torch
import torchaudio


class AudioLoader:
    def __init__(self, desired_sample_rate: int = 16000):
        self.desired_sample_rate = desired_sample_rate

    def load_audio(self, path: str):
        """
        Load audio from a file and optionally resample it if necessary.
        The preferred sampling rate is 16kHz due to the sampling rate of audio
        data on which the model has been trained.

        Args:
            path (str): Path to the audio file to load.
        """
        try:
            waveform, sample_rate = torchaudio.load(path)

            if sample_rate != self.desired_sample_rate:
                waveform = self._resample_waveform(waveform, sample_rate)
            return waveform

        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {path}")
        except Exception as e:
            raise Exception(f"An error occurred while loading audio: {e}")

    def _resample_waveform(self, waveform: torch.Tensor, sample_rate: int):
        """
        Resample waveform to 16kHz, this is the sample rate preferred for forced
        alignment as the model was trained on audio samples with that sampling rate.

        Args:
            desired_sample_rate (int): Desired sample rate for waveform.

        Returns:
            torch.Tensor: Resampled waveform.
        """
        try:
            transform = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=self.desired_sample_rate
            )
            return transform(waveform)
        except Exception as e:
            raise Exception(f"An error occurred during resampling: {e}")
