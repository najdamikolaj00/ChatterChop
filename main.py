from chatterchop.chop import ChatterChop
import os

def main():
    audio_path = os.path.join('test_data/test_audio_pl_shorter.wav')
    output_path = os.path.join('test_data/test_split_short/HC1')
    obj1 = ChatterChop(audio_path)
    obj1.chop_chatter()
    obj1.save_speech_segments(output_path)

if __name__ == '__main__':
    main()