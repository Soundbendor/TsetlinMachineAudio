import os
from pydub import AudioSegment
import pydub
import math
import re

def process_audio(input_file):
    # Check the file is ok to use
    file_pattern = '_[aeiou]\.wav'
    match = re.search(file_pattern, input_file)
    if match is None:
        return
    # Remove silence from beginning and end    
    sound = AudioSegment.from_wav(input_file)
    sound = sound.strip_silence(silence_len=100, silence_thresh=-60, padding=40)

    file_no_ext = input_file[:-4]
    file_label = input_file[-5]
    # Split into nearly identical segments
    segments = sound[::2000]  # 2000 milliseconds = 2 seconds

    # Process each segment
    for i, segment in enumerate(segments):
        
        # Number of frames in segment:
        num_frames = segment.frame_count()

        # skip segment if too small (~1.66s)
        if num_frames <= 73200:
            pass
        # Pad with silence if segment is almost long enough
        elif num_frames < 88200:
            # Silent segment
            silence_duration = (88200 - num_frames) / 44100 # Calculate the required silent duration
            silence = AudioSegment.silent(duration=silence_duration, frame_rate=44100)
            segment += silence
            
            #assert int(segment.frame_count()) == 88200 , f"Padding failed: Frames = {segment.frame_count()}, input = {num_frames}, curr_file: {input_file}, silence duration = {silence.frame_count()}"
            

            # Assuming this worked, I need to spit out files
            segment.export(f"{file_no_ext}_{i}_{file_label}.wav", format="wav")
            
        # If segment is 2 seconds long
        elif int(num_frames) == 88200:
            segment.export(f"{file_no_ext}_{i}_{file_label}.wav", format="wav")

        # If segment needs to be trimmed down to 2 seconds
        elif int(num_frames) > 88200:
            # Remove last ms
            trimmed = segment[:1999]
            trim_frames = trimmed.frame_count()
            # add difference from 2 seconds as silence
            silence_duration = (88200 - trim_frames) / 44100  # Calculate the required silent duration
           
            silence = AudioSegment.silent(duration=silence_duration, frame_rate=44100)
            trimmed += silence
    
            #assert int(trimmed.frame_count()) == 88200, f"Trimming failed: Frames = {trimmed.frame_count()}, input = {num_frames}, curr_file: {input_file}"

            trimmed.export(f"{file_no_ext}_{i}_{file_label}.wav", format="wav")

        else:
            raise AttributeError("Unknown error. Check number of frames, format, etc.")
            break

def process_directory(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".wav"):
                process_audio(os.path.join(root, file))

def main():
    current_directory = os.getcwd()
    process_directory(current_directory)

if __name__ == "__main__":
    choice = input("Are you sure you want to do this? It copies almost all of the .wav files it finds. [Y/n]")
    if choice not in ["Y","n"]:
        print("Invalid option. Exiting Now.")
        exit() # sys.exit?
    elif choice == "n":
        print("Exiting.")
        exit()    
    elif choice == "Y":
        main()
    else:
        exit()
