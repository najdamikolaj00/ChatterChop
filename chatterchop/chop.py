"""Main ChopChatter class and all the corresponding methods."""

from .transcription import (
    whisper_transcription
)

from .forced_alignment import (
    get_word_segments
)

class ChatterChop:
    def __init__(self, audio_path: str=None, output_path: str=None):
        pass