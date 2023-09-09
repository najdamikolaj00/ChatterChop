"""This file contains tutorial for all of the methods provided by a package."""

import sys
sys.path.append(r'd:\ChatterChop')

from chatterchop.chop import ChatterChop
import os

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

    speech_transcription()