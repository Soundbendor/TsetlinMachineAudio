import numpy as np

from librosa.feature import mfcc as MFCC
from sklearn.preprocessing import KBinsDiscretizer
from pydub import AudioSegment
from sklearn.utils import shuffle
import pickle
import re
import os
import json
import datetime
import shutil
from pydub.generators import WhiteNoise


def get_save_path(args, HEAD):
    """Make save path
    """
    date = '{}'.format(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'))
    suffix = "{}_{}_{}_{}.pickle".format(args[0], args[1], args[2], date)
    result_path = os.path.join(HEAD, suffix)
    return result_path


def convert_to_binary(array):
    max_value = np.max(array)
    num_bits = int(np.ceil(np.log2(max_value + 1)))  # Determine number of bits needed
    binary_array = np.zeros((array.shape[0], array.shape[1] * num_bits), dtype=int)
    
    # Calculate powers of 2 to vectorize the conversion
    powers_of_2 = 2 ** np.arange(num_bits - 1, -1, -1)
    
    # Compute binary representation using vectorized operations
    for i in range(num_bits):
        binary_array[:, i::num_bits] = (array >> (num_bits - i - 1)) & 1
    
    return binary_array


def shrink_to_1_1(x, bit_depth):
    try:
        iterator = iter(x)
    except TypeError:
        return x / 2 ** bit_depth
    else:
        x_list = []
        for i in x:
            x_list.append(i / 2 ** bit_depth)
        return x_list


def gen_mfccs(x, config):
    mfcc_list = []
    for arr in x:
        mfccs = MFCC(y=arr,  # librosa calls input to mfcc y
                     sr=config["sample_rate"],
                     n_mfcc=config["n_mfcc"],
                     n_fft=config["n_fft"],
                     hop_length=config["hop_length"],
                     win_length=config["win_length"],
                     n_mels=config["n_mels"],
                     center=False)
        if config["avg_mfccs"] == True:
            mean_mfccs = np.mean(mfccs, axis=1)  # TODO correct axis?
            mfcc_list.append(mean_mfccs)
        else:
            mfcc_list.append(mfccs)
    return mfcc_list


def booleanize(x, booleanizer, config, train=True):
    if type(x) == list:
        bool_list = []
        if train:
            for mfcc_vector in x:
                x_bools = booleanizer.fit_transform(mfcc_vector.T)
                bool_list.append(x_bools)
            return bool_list
        else:
            for mfcc_vector in x:
                x_bools = booleanizer.transform(mfcc_vector.T)
                bool_list.append(x_bools)
            return bool_list

    else:
        if len(x.shape) > 2:
            n, m, t = x.shape
            x = x.transpose(0, 2, 1)
            x = x.reshape(n * t, m)
            if train:
                x_bools = booleanizer.fit_transform(x)
                x_bools = x_bools.reshape(n, -1)
            else:
                x_bools = booleanizer.transform(x)
                x_bools = x_bools.reshape(n, -1)
        elif len(x.shape) == 2:
            if train:
                x_bools = booleanizer.fit_transform(x)
            else:
                x_bools = booleanizer.transform(x)
    return convert_to_binary(x_bools.astype(np.uint8)).astype("?")
    #return x_bools.astype("?")


def process_audio(input_file, config, verbose=False):
    valid_class_types = {"vowel", "singer", "technique","all"}
    class_type = config["class_type"]
    if class_type not in valid_class_types:
        raise ValueError("results: class_type must be one of %r." % valid_class_types)

    # Check the file is ok to use
    if not input_file.endswith(".wav"):
        raise ValueError(".wav file not found")

    # assign labels based on class type


    file_pattern = r'_([aeiou]).wav'
    match = re.search(file_pattern, input_file)
    if match:
        if verbose:
            print(f"Pattern match found: {match.group(1)}")
        vowel = match.group(1)
        vowel_to_class = {'a': 0, 'e': 1, 'i': 2, 'o': 3, 'u': 4}
        vowel_label = vowel_to_class.get(vowel)
    else:
        vowel_label = -1

    file_pattern = r'(male|female)([0-9][0-1]?|[0-9][0-1]?)'
    match = re.search(file_pattern, input_file)
    # male [0-10], female[11-19]
    if match:
        if verbose:
            print(f"Pattern match found: {match.group(1)}")
        sex_to_class = {'male': 0, 'female': 1}
        sex = sex_to_class[match.group(1)]
        if sex == 0:
            id = int(match.group(2)) - 1
        else:
            id = int(match.group(2)) + 10
        singer_label = id
    else:
        singer_label = -1

    file_pattern = r'\b(vibrato|straight|breathy|vocal_fry|lip_trill|trill|trillo|inhaled|belt|spoken)\b'
    match = re.search(file_pattern, input_file)
    if match:
        if verbose:
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
        tech_label = tech_to_class[match.group(1)]
    else:
        tech_label = -1

    label = (vowel_label,tech_label,singer_label)

    # Remove silence from beginning and end
    sound = AudioSegment.from_wav(input_file)
    sound = sound.strip_silence(silence_len=100, silence_thresh=-60, padding=40)
    sound = sound.set_frame_rate(config["sample_rate"])
    if config["add_noise"]:
        noise = WhiteNoise(sample_rate=config["sample_rate"]).to_audio_segment(duration=len(sound))
        sound = sound.overlay(noise)
    seg_length_ms = config["seg_length"] // config["sample_rate"] * 1000
    # Split into nearly identical segments
    segments = sound[::seg_length_ms]
    processed_segments = []
    labels = []
    # Process each segment
    for i, segment in enumerate(segments):


        # Number of frames in segment:
        num_frames = segment.frame_count()

        # skip segment if too small 3/4 is the cutoff
        if num_frames <= int(3 / 4 * config["seg_length"]):
            if verbose:
                print(f"Segment {i} discarded: {num_frames} less than {int(3 / 4 * config['seg_length'])}")
            pass



        # Pad with silence if segment is almost long enough
        elif num_frames < config["seg_length"]:
            if verbose:
                print(f"Segment {i} too small, padding length: {num_frames} / {config['seg_length']}")
            # convert to numpy:
            seg_array = np.array(segment.get_array_of_samples(), dtype=np.float32)
            frames_needed = int(config["seg_length"] - num_frames)  # convert from float
            padded_seg = np.pad(seg_array, (0, frames_needed), mode='constant')
            assert padded_seg.shape[0] == config[
                "seg_length"], f"Padding failed: Frames = {padded_seg.sahpe[0]}, input = {num_frames}, curr_file: {input_file}"

            processed_segments.append(padded_seg)
            labels.append(label)

        # If segment is exactly correct
        elif int(num_frames) == config["seg_length"]:
            if verbose:
                print(f"Segment {i} length correct. Length: {num_frames}")
            processed_segments.append(np.array(segment.get_array_of_samples(), dtype=np.float32))
            labels.append(label)

        # If segment needs to be trimmed down
        elif int(num_frames) > config["seg_length"]:
            if verbose:
                print(f"Segment {i} too large. Length: {num_frames}. Trimming.")

            # convert to numpy
            seg_array = np.array(segment.get_array_of_samples(), dtype=np.float32)
            trimmed = seg_array[:config["seg_length"]]

            assert int(trimmed.shape[0]) == config[
                "seg_length"], f"Trimming failed: Frames = {trimmed.shape[0]}, input = {num_frames}, curr_file: {input_file}"

            processed_segments.append(trimmed)
            labels.append(label)
        else:
            raise AttributeError("Unknown error. Check number of frames, format, etc.")
        if len(processed_segments) > 0:
            assert processed_segments[-1].shape[0] == config[
                "seg_length"], f"Most recent segment wrong length: {processed_segments[-1].shape[0]}"

    return processed_segments, labels


# @profile
def process_directory(directory, booleanizer, config, train=True, bool_mode=True, verbose=False):
    x_out = []
    y_out = []
    file_count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".wav"):
                file_count += 1

                result = process_audio(os.path.join(root, file), config, verbose=verbose)
                if result is None:
                    continue
                else:
                    x, y = result
                x = shrink_to_1_1(x, config["bit_depth"])
                mfccs = gen_mfccs(x, config)
                if (config["delay_bools"] == False) and bool_mode:
                    x_bools = booleanize(mfccs.T, booleanizer, config, train)
                    x_out += x_bools
                    y_out += y
                else:
                    x_out += mfccs
                    y_out += y

    print(f"Files processed: {file_count}")
    if (config["delay_bools"] is False) and bool_mode:
        return x_out, y_out
    elif config["delay_bools"] and bool_mode:
        x_out = np.array(x_out)  # overwrite to save memory
        x_out = booleanize(x_out, booleanizer, config)
        return x_out, y_out
    elif not bool_mode:
        x_out = np.array(x_out)
        if len(x_out.shape) > 2:
            n, m, t = x_out.shape
            x_out = x_out.reshape(n, -1)
        return x_out, y_out  # As MFCCs


def main():
    current_directory = os.getcwd()
    with open("config_npy.json", 'r') as f:
        config = json.load(f)

    booleanizer = KBinsDiscretizer(n_bins=config["num_quantiles"], encode=config["boolean_encoding"])

    # First do the training set.
    TRAIN_DATA_PATH = config["train_directory"]
    X, Y = process_directory(TRAIN_DATA_PATH, booleanizer, config, bool_mode=config["return_bools"], verbose=False)

    x_file_path = get_save_path([config["class_type"], "X", config["fold"]], config["data_out_path"])
    y_file_path = get_save_path([config["class_type"], "y", config["fold"]], config["data_out_path"])

    if type(X) == list:
        X = np.vstack(X)
    Y = np.vstack(Y)

    X, Y = shuffle(X, Y)

    #np.save(x_file_path, X)
    #np.save(y_file_path, Y)

    print(f"Training data processed: final shape of training X: {X.shape} and Y: {Y.shape}")

    # Next the Test set using the same statistics as the train. (for booleanizer)
    #TEST_DATA_PATH = config["test_directory"]
    #test_X, test_Y = process_directory(TEST_DATA_PATH, booleanizer, config, train=False, bool_mode=config["return_bools"], verbose=True)

    test_x_file_path = get_save_path([config["class_type"], "X_test", config["fold"]], config["data_out_path"])
    test_y_file_path = get_save_path([config["class_type"], "y_test", config["fold"]], config["data_out_path"])

    #if test_X is not None:
    #    if type(test_X) == list:
    #        test_X = np.vstack(test_X)
    #    test_y = np.vstack(test_Y)
#
#        test_X, test_y = shuffle(test_X, test_y)

        #np.save(test_x_file_path, test_X)
        #np.save(test_y_file_path, test_y)

    #data_dict = {"x_train": X,"x_test": test_X, "y_train" : Y, "y_test": test_y}
    data_dict = {"x": X, "y":Y}
    with open(get_save_path([config["class_type"], "all", config["fold"]], config["data_out_path"]), "wb") as f:
        pickle.dump(data_dict,f)

    log_name = os.path.join(config["data_out_path"], "log{}".format(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')))
    shutil.copyfile("config_npy.json", log_name)


if __name__ == "__main__":
    main()
