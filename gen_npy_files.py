import numpy as np
from tqdm import tqdm
import librosa
from librosa.feature import mfcc as MFCC
from sklearn.preprocessing import KBinsDiscretizer
from pyTsetlinMachine.tm import MultiClassTsetlinMachine
from pydub import AudioSegment
import re
import os
import json
import datetime
import shutil

def get_save_path(args, HEAD):
    """Make save path
    """
    date = '{}'.format( datetime.datetime.now().strftime('%Y-%m-%d-%H-%M') )
    suffix = "{}_{}_{}".format(args[0], args[1], date)
    result_path = os.path.join(HEAD, suffix)
    return result_path


#TODO consider downsampling
#TODO Add log file

def shrink_to_1_1(x,bit_depth):
    try:
        iterator = iter(x)
    except TypeError:
        return x/2**bit_depth
    else:
        x_list = []
        for i in x:
            x_list.append(i/2**bit_depth)
        return x_list

def gen_mfccs(x,config):
    mfcc_list = []
    for arr in x:
        mfccs = MFCC(y=arr, # librosa calls input to mfcc y
                        sr=config["sample_rate"],
                        n_mfcc=config["n_mfcc"],
                        n_fft=config["n_fft"],
                        hop_length=config["hop_length"],
                        win_length=config["win_length"],
                        n_mels=config["n_mels"],
                        center=False)
        if config["average_mfccs"] == True:
            #TODO implement
            pass
        else:
            mfcc_list.append(mfccs)
    return mfcc_list

def booleanize(x,booleanizer,config):
    # TODO booleanize only at very end of data? KWS paper says "[MFCC vector] can then be passed to booleanizer module"
    # Which implies it is done per vector, by which I assume it means per file (since the "vector" has a time dimension)
    if type(x) == list:
        bool_list = []
        for mfcc_vector in x:
            x_bools = booleanizer.fit_transform(mfcc_vector.T)
            bool_list.append(x_bools)
        return bool_list
    else:
        x_bools = booleanizer.fit_transform(x.transpose(0,2,1)) 
    
    return x_bools




def process_audio(input_file,config):
    valid_class_types = {"vowel","singer","technique"}
    class_type = config["class_type"]
    if class_type not in valid_class_types:
      raise ValueError("results: class_type must be one of %r." % valid_class_types)

    # Check the file is ok to use
    if not input_file.endswith(".wav"):
      raise ValueError(".wav file not found")

    # assign labels based on class type
    match class_type:
      case "vowel":
        file_pattern = r'_([aeiou]).wav'
        match = re.search(file_pattern, input_file)
        if match:
            print(f"Pattern match found: {match.group(1)}")
            vowel = match.group(1)
            vowel_to_class = {'a': 0, 'e': 1, 'i': 2, 'o': 3, 'u': 4}
            label = vowel_to_class.get(vowel)

      case "singer":
        file_pattern = r'(male|female)([0-9][0-1]?|[0-9][0-1]?)'
        match = re.search(file_pattern, input_file)
        # male [0-10], female[11-19]
        if match:
            print(f"Pattern match found: {match.group(1)}")
            sex_to_class = {'male':0,'female':1}
            sex = sex_to_class[match.group(1)]
            if sex == 0:
              id = int(match.group(2))-1
            else:
              id = int(match.group(2))+10
            label = (sex, id)

      case "technique":
        file_pattern = r'(vibrato|straight|breathy|vocal_fry|lip_trill|trill|trillo|inhaled|belt|spoken)'
        match = re.search(file_pattern, input_file)
        if match:
          print(f"Pattern match found: {match.group(1)}")
          tech_to_class = {
              'vibrato': 0,
              'straight': 1,
              'breathy': 2,
              'vocal_fry': 3,
              'lip_trill': 4,
              'trill': 5,
              'trillo': 6,
              'inhaled': 7,
              'belt': 8,
              'spoken': 9
          }
          label = tech_to_class[match.group(1)]


    # Remove silence from beginning and end
    sound = AudioSegment.from_wav(input_file)
    sound = sound.strip_silence(silence_len=100, silence_thresh=-60, padding=40)
    sound = sound.set_frame_rate(config["sample_rate"])

    seg_length_ms = config["seg_length"]//config["sample_rate"] * 1000
    # Split into nearly identical segments
    segments = sound[::seg_length_ms]  
    processed_segments = []
    labels = []
    # Process each segment
    for i, segment in enumerate(segments):

        # Number of frames in segment:
        num_frames = segment.frame_count()

        # skip segment if too small 3/4 is the cutoff
        if num_frames <= int(3/4 * config["seg_length"]):
            print(f"Segment {i} discarded: {num_frames} less than {int(3/4 * config['seg_length'])}")
            pass



        # Pad with silence if segment is almost long enough
        elif num_frames < config["seg_length"]:
            print(f"Segment {i} too small, padding length: {num_frames} / {config['seg_length']}")
            # convert to numpy:
            seg_array = np.array(segment.get_array_of_samples(),dtype=np.float32)
            frames_needed = int(config["seg_length"] - num_frames) # convert from float
            padded_seg = np.pad(seg_array,(0,frames_needed),mode='constant')
            assert padded_seg.shape[0] == config["seg_length"] , f"Padding failed: Frames = {padded_seg.sahpe[0]}, input = {num_frames}, curr_file: {input_file}"
            
            processed_segments.append(padded_seg)

        # If segment is exactly correct
        elif int(num_frames) == config["seg_length"]:
            print(f"Segment {i} length correct. Length: {num_frames}")
            processed_segments.append(np.array(segment.get_array_of_samples(),dtype=np.float32))

        # If segment needs to be trimmed down
        elif int(num_frames) > config["seg_length"]:
            print(f"Segment {i} too large. Length: {num_frames}. Trimming.")
            
            #convert to numpy
            seg_array = np.array(segment.get_array_of_samples(),dtype=np.float32)
            trimmed = seg_array[:config["seg_length"]]
            
            assert int(trimmed.shape[0]) == config["seg_length"], f"Trimming failed: Frames = {trimmed.shape[0]}, input = {num_frames}, curr_file: {input_file}"

    
            processed_segments.append(trimmed)
        else:
            raise AttributeError("Unknown error. Check number of frames, format, etc.")
        if len(processed_segments) > 0:
          assert processed_segments[-1].shape[0] == config["seg_length"], f"Most recent segment wrong length: {processed_segments[-1].shape[0]}"

        # if no error raised, all if branches get a label, and all labels are identical
        labels.append(label)
    return processed_segments, labels


def process_directory(directory, booleanizer, config):
    x_out = []
    y_out = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".wav"):
                x, y = process_audio(os.path.join(root, file), config)
                resized_x = shrink_to_1_1(x,config["bit_depth"])
                mfccs = gen_mfccs(resized_x,config)
                if config["delay_bool"] == False:
                    x_bools = booleanize(mfccs.T, booleanizer, config)
                    x_out += x_bools
                    y_out += y
                else:
                    x_out += mfccs
                    y_out += y
               
                
    if config["delay_bool"] == False:
        return x_out,y_out
    else:
        mfcc_matrix = np.array(x_out)
        bool_matrix = booleanize(mfcc_matrix,booleanizer,config)
        return bool_matrix, y
                




def main():
    current_directory = os.getcwd()
    with open("config_npy.json", 'r') as f:
        config = json.load(f)


    
  
    booleanizer = KBinsDiscretizer(n_bins=config["num_quantiles"],encode=config["boolean_encoding"])

    DATA_PATH = config["data_path"]
    X, Y = process_directory(DATA_PATH,booleanizer,config)
    


    x_file_path = get_save_path([config["class_type"],"X"],config["data_out_path"])
    y_file_path = get_save_path([config["class_type"],"y"],config["data_out_path"])
    log_name = os.path.join(config["data_out_path"],"log{}".format( datetime.datetime.now().strftime('%Y-%m-%d-%H-%M') )")
    shutil.copyfile("config_npy.json",log_name)
    

    if type(X) == list:
        x_mat = np.vstack(X)
    y_mat = np.vstack(Y)


    np.save(x_file_path,x_mat)
    np.save(y_file_path,y_mat)




if __name__ == "__main__":
    choice = input("Are you sure you want to do this? It generates files. [Y/n]")
    if choice not in ["Y","n"]:
        print("Invalid option. Exiting Now.")
        exit() # sys.exit?
    elif choice == "n":
        print("Exiting.")
        exit()    
    elif choice == "Y":
        main()
