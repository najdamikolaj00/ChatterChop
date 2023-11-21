"""This file contains tutorial for all of the methods provided by a package."""

import sys
from pathlib import Path

from chatterchop import WhisperTranscriptor

sys.path.append(r"d:\ChatterChop")
from chatterchop.ChatterChopper import ChatterChopper
import os


def speech_chop():
    """
    When the path to audio is provided without forced any transcription file
    the transcription is performed using OpenAI Whisper.

    1st Example (Polish):
        The object is called with the path to the audio file. Then the audio is cut and
    segments to the desired directory. New audio segments are identified by
    and the word position in an utterance.

    test_audio_path_pl - path to Polish audio file.
    output_dir_pl - path to output directory.

    2nd Example (English):
        The object is called with the path to the audio file. Then the audio is cut and
    segments to the desired directory. New audio segments are identified by
    and the word position in an utterance.

    test_audio_path_eng - path to English audio file.
    output_dir_eng - path to output directory.

    """
    # 1st Example (PL):
    # test_audio_path_pl = 'data_to_test/test_pl/test_audio_pl.wav'
    # output_dir_pl = 'data_to_test/test_pl/test_split_pl'

    # test_obj_pl = ChatterChop(test_audio_path_pl)

    # test_obj_pl.chop_chatter()

    # test_obj_pl.save_speech_segments(output_dir_pl)

    # 2nd Example (ENG):
    test_audio_path_eng = "data_to_test/test_eng/test_audio_eng.wav"
    output_dir_eng = "data_to_test/test_eng/test_split_eng"

    chatter_chopper = ChatterChopper(
        whisper_transcriptor=WhisperTranscriptor(language="en")
    )

    chopped_chatter = chatter_chopper.chop_chatter(test_audio_path_eng)

    chopped_chatter.save(output_dir_eng)


def speech_transcription():
    """
    When the path to audio is provided without forced any transcription file
    the transcription is performed using OpenAI Whisper.

    1st Example:
        The object is called with path to audio file. Then accuracy and
    saving transcription to a file is performed.

    #2nd Example:
        The object is called with path to audio file and to transcript.
    Then accuracy is performed.

    #3rd Example:
        The object is called with path to audio file and to transcript.
    And the ground truth is provided in as a string. Then accuracy is performed.

    test_audio_path - path to audio file.
    test_ground_truth_path - path to test ground truth transcription file.
    test_transcription_file - path to transcript.
    test_obj - ChatterChop object.

    """

    test_audio_path = "data_to_test/test_eng/test_audio_eng.wav"
    test_ground_truth_path = Path(
        "data_to_test/test_eng/test_transcription_ground_truth_eng.txt"
    )
    test_transcription_file = "data_to_test/test_eng/test_transcription_eng.txt"
    # 1st Example:
    chatter_chopper = ChatterChopper()
    test_obj_1 = chatter_chopper.chop_chatter(test_audio_path)

    transcription_result = test_obj_1.get_transcription_accuracy(test_ground_truth_path)
    print(transcription_result)

    test_obj_1.save_transcription("data_to_test/test_eng/saved_trans_eng.txt")

    # 2nd Example:
    # test_obj_2 = ChatterChop(test_audio_path, test_transcription_file)

    # transcription_result = test_obj_2.get_transcription_accuracy(test_ground_truth_path)
    # print(transcription_result)

    # 3nd Example:
    # test_obj_3 = ChatterChop(test_audio_path, test_transcription_file)

    # test_ground_truth = 'Warszawa jest pełnym sprzeczności przez wielu niezniszczalnym.'
    # transcription_result = test_obj_3.get_transcription_accuracy(test_ground_truth)
    # print(transcription_result)


if __name__ == "__main__":
    """
    Uncomment any of these functions
    to check their capabilities.

    """

    # speech_transcription()

    speech_chop()
