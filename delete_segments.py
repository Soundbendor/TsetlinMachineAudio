import os
import re

def remove_segment(file):
    assert file[-4:] == ".wav"
    match = re.search("_[aeiou]_[0-9]_[aeiou]\.wav", file)
    if match:
        if os.path.exists(file):
            os.remove(file)
        else:
            raise FileExistsError("Path does not exist.")


def process_directory_rm(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".wav"):
                remove_segment(os.path.join(root, file))


def main():
    current_directory = os.getcwd()
    process_directory_rm(current_directory)

if __name__ == "__main__":
    main()