"""This file contains functions to perform transcript alignment to audio-speech
functions written based on:  https://pytorch.org/audio/stable/tutorials/forced_alignment_tutorial.html

About CTC-segmentation:

Kürzinger, Ludwig, et al. "CTC-segmentation of large corpora for german end-to-end speech recognition." 
International Conference on Speech and Computer. Cham: Springer International Publishing, 2020.
"""

import os
from dataclasses import dataclass
from typing import Sequence

import torch
import torchaudio

from .utils import check_cuda_availability, normalize_transcript_CTC

DEVICE = check_cuda_availability()


def load_bundle() -> tuple[torch.nn.Module, tuple[str, ...]]:
    """
    Function description

    Returns:
        model: Module.
        Tuple[str]: labels.

    """
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model().to(DEVICE)
    labels = bundle.get_labels()

    return model, labels


def generate_frame_wise_probability(
    waveform: torch.Tensor, model: torch.nn.Module
) -> torch.Tensor:
    """
    Function description

    Args:
        waveform (torch.Tensor): Waveform of speech audio.

    Returns:
        model: Module.
        torch.Tensor: Frame wise probability.

    """

    with torch.inference_mode():
        emissions, _ = model(waveform.to(DEVICE))
        emissions = torch.log_softmax(emissions, dim=-1)

    emission = emissions[0].cpu().detach()

    return emission


def get_tokens(transcript: str, labels: Sequence[str]) -> list[int]:
    """
    Convert a transcript into a sequence of token indices using a given label dictionary.

    Args:
        transcript (str): The transcript to convert into tokens.
        labels (list): A list of label characters.

    Returns:
        tokens (list): A list of token indices.
    """
    dictionary = {c: i for i, c in enumerate(labels)}
    tokens = [dictionary[c] for c in transcript]
    return tokens


def generate_alignment_probability(
    emission: torch.Tensor, tokens: list[int], blank_id: int = 0
) -> torch.Tensor:
    """
    Generate an alignment probability matrix using Viterbi algorithm.

    Args:
        emission (torch.Tensor): Frame-wise emission probabilities.
        tokens (list): A list of token indices.
        blank_id (int): Index of the blank token.

    Returns:
        trellis (torch.Tensor): The alignment probability trellis.
    """
    num_frame = emission.size(0)
    num_tokens = len(tokens)
    trellis = torch.empty((num_frame + 1, num_tokens + 1))
    trellis[0, 0] = 0
    trellis[1:, 0] = torch.cumsum(emission[:, 0], 0)
    trellis[0, -num_tokens:] = -float("inf")
    trellis[-num_tokens:, 0] = float("inf")
    for t in range(num_frame):
        trellis[t + 1, 1:] = torch.maximum(
            trellis[t, 1:] + emission[t, blank_id],
            trellis[t, :-1] + emission[t, tokens],
        )
    return trellis


@dataclass
class Point:
    """
    Represents a point in the alignment path.

    Attributes:
        token_index (int): Index of the token.
        time_index (int): Index of the time/frame.
        score (float): Score associated with the alignment point.
    """

    token_index: int
    time_index: int
    score: float


def backtrack(
    trellis: torch.Tensor, emission: torch.Tensor, tokens: list[int], blank_id: int = 0
) -> list[Point]:
    """
    Backtrack through the alignment trellis to find the optimal alignment path.

    Args:
        trellis (torch.Tensor): The alignment probability trellis.
        emission (torch.Tensor): Frame-wise emission probabilities.
        tokens (list): A list of token indices.
        blank_id (int): Index of the blank token.

    Returns:
        path (list): The optimal alignment path represented as a list of Point objects.
    """
    j = trellis.size(1) - 1
    t_start = torch.argmax(trellis[:, j]).item()
    path = []
    for t in range(t_start, 0, -1):
        stayed = trellis[t - 1, j] + emission[t - 1, blank_id]
        changed = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]
        prob = emission[t - 1, tokens[j - 1] if changed > stayed else 0].exp().item()
        path.append(Point(j - 1, t - 1, prob))
        if changed > stayed:
            j -= 1
            if j == 0:
                break
    else:
        raise ValueError("Failed to align")
    return path[::-1]


@dataclass
class Segment:
    """
    Represents a segment of the alignment with label, start, end, and score.

    Attributes:
        label (str): Label of the segment.
        start (int): Start index of the segment.
        end (int): End index of the segment.
        score (float): Score associated with the segment.

    Methods:
        __repr__: Returns a string representation of the segment.
        length: Returns the length of the segment.
    """

    label: str
    start: int
    end: int
    score: float

    def __repr__(self):
        return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"

    @property
    def length(self):
        return self.end - self.start


def merge_repeats(path: list[Point], transcript: str) -> list[Segment]:
    """
    Merge repeated tokens in the alignment path to create segments.

    Args:
        path (list): The alignment path represented as a list of Point objects.
        transcript (str): The original transcript.

    Returns:
        segments (list): A list of Segment objects representing merged segments.
    """
    i1, i2 = 0, 0
    segments = []
    while i1 < len(path):
        while i2 < len(path) and path[i1].token_index == path[i2].token_index:
            i2 += 1
        score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
        segments.append(
            Segment(
                transcript[path[i1].token_index],
                path[i1].time_index,
                path[i2 - 1].time_index + 1,
                score,
            )
        )
        i1 = i2
    return segments


def merge_words(segments: list[Segment], separator: str = "|") -> list[Segment]:
    """
    Merge consecutive segments with the same label into words.

    Args:
        segments (list): A list of Segment objects.
        separator (str): Separator to use between merged segments.

    Returns:
        words (list): A list of merged word segments.
    """
    words = []
    i1, i2 = 0, 0
    while i1 < len(segments):
        if i2 >= len(segments) or segments[i2].label == separator:
            if i1 != i2:
                segs = segments[i1:i2]
                word = "".join([seg.label for seg in segs])
                score = sum(seg.score * seg.length for seg in segs) / sum(
                    seg.length for seg in segs
                )
                words.append(
                    Segment(word, segments[i1].start, segments[i2 - 1].end, score)
                )
            i1 = i2 + 1
            i2 = i1
        else:
            i2 += 1
    return words


def run_forced_alignment(
    waveform: torch.Tensor, transcript_file: str
) -> tuple[list[Segment], int]:
    """
    Run forced alignment on an audio waveform with a transcript.

    Args:
        waveform (torch.Tensor): The audio waveform data.
        transcript_file (str): Path to the transcript file or the transcript text.

    Returns:
        word_segments (list): A list of word segments with timing information.
        trellis_size (int): The size of the alignment trellis.
    """
    model, labels = load_bundle()
    emission = generate_frame_wise_probability(waveform, model)

    if os.path.isfile(transcript_file):
        with open(transcript_file, "r", encoding="utf-8") as f:
            transcript = f.read()
    else:
        transcript = transcript_file

    normalized_transcript = normalize_transcript_CTC(transcript)
    tokens = get_tokens(normalized_transcript, labels)
    trellis = generate_alignment_probability(emission, tokens)
    path = backtrack(trellis, emission, tokens)
    segments = merge_repeats(path, normalized_transcript)
    word_segments = merge_words(segments)

    return word_segments, trellis.size(0)
