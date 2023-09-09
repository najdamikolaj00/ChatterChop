"""This file contains tutorial for all of the methods provided by a package."""

import sys
sys.path.append(r'd:\ChatterChop')
from chatterchop.chop import ChatterChop
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
    #1st Example:
    test_audio_path_pl = 'data_to_test/test_pl/test_audio_pl.wav'
    output_dir_pl = 'data_to_test/test_pl/test_split_pl'

    test_obj_pl = ChatterChop(test_audio_path_pl)

    test_obj_pl.chop_chatter()

    test_obj_pl.save_speech_segments(output_dir_pl)

    #2nd Example:
    test_audio_path_eng = 'data_to_test/test_eng/test_audio_eng.wav'
    output_dir_eng = 'data_to_test/test_eng/test_split_eng'

    test_obj_eng = ChatterChop(test_audio_path_eng)

    test_obj_eng.chop_chatter()

    test_obj_eng.save_speech_segments(output_dir_eng)

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
    
    test_audio_path = 'data_to_test/test_audio_pl_shorter.wav'
    test_ground_truth_path = 'data_to_test/test_transcription_ground_truth.txt'
    test_transcription_file = 'data_to_test/test_transcription.txt'
    #1st Example:
    # test_obj_1 = ChatterChop(test_audio_path)


    # transcription_result = test_obj_1.get_transcription_accuracy(test_ground_truth_path)
    # print(transcription_result)

    # test_obj_1.save_transcription('data_to_test/saved_trans.txt')


    #2nd Example:
    # test_obj_2 = ChatterChop(test_audio_path, test_transcription_file)


    # transcription_result = test_obj_2.get_transcription_accuracy(test_ground_truth_path)
    # print(transcription_result)

    #3nd Example:
    test_obj_3 = ChatterChop(test_audio_path, test_transcription_file)

    test_ground_truth = 'Warszawa jest pełnym sprzeczności przez wielu niezniszczalnym.'
    transcription_result = test_obj_3.get_transcription_accuracy(test_ground_truth)
    print(transcription_result)

if __name__ == '__main__':
    """
    Uncomment any of these functions 
    to check their capabilities.

    """

    #speech_transcription()

    speech_chop()