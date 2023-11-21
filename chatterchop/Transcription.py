from pathlib import Path

from chatterchop.utils import cer_metric, wer_metric


class Transcription(str):
    def get_accuracy(
        self, ground_truth: str | Path, verbose: bool = False
    ) -> tuple[float, float]:
        """
        Returns transcription accuracy based on WER and CER metric.
            Args:
                ground_truth (str): Path to file containing ground truth.
                verbose: Specifies it the result should be printed

            Returns:
                WER and CER results in %

            Prints:
                WER and CER results in % or None if isn't verbose.


        """
        if isinstance(ground_truth, Path):
            ground_truth = ground_truth.read_text()
        wer = wer_metric(self, ground_truth)
        cer = cer_metric(self, ground_truth)
        if verbose:
            print(f"WER: {wer * 100:.2f}%, CER:{cer * 100:.2f}%")
        return wer, cer
