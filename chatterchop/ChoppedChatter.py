import os
from pathlib import Path

import torchaudio

from chatterchop.Transcription import Transcription
from chatterchop.utils import transcription_into_txt


class ChoppedChatter(list):
    def __init__(self, sample_rate: int, transcription: Transcription):
        super().__init__()
        self.sample_rate = sample_rate
        self.transcription = transcription

    def save(self, output_path: str) -> None:
        """
        Saves chopped audio in desired path using
        recognised/transcripted word in uppercase as a filename.

        Args:
            output_path (str): Path to the specific output directory.

        """
        for i, segment in enumerate(self):
            torchaudio.save(
                os.path.join(output_path, f'{segment["word"]}_{i}.wav'),
                segment["audio_segment"],
                sample_rate=self.sample_rate,
            )

    def get_transcription_accuracy(self, ground_truth: str | Path):
        return self.transcription.get_accuracy(ground_truth)

    def save_transcription(self, txt_path):
        transcription_into_txt(self.transcription, txt_path)
