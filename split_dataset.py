import os #used for file operation , naviagte, join paths
import shutil #used for copying files from source to destination
import random #used for shuffling the list of images to ensure random distribution across train, val, test sets

random.seed(42) #this is randomness value

source_real = "CASIA2/Au"  #where the real images are stored
source_fake = "CASIA2/Tp" #where the fake images are stored

base_dir = "dataset" # the new home for our organized dataset

def split_and_copy(source_folder, class_name):
    images = os.listdir(source_folder) # This gets all file names inside folder.
    random.shuffle(images) #shuffle images

    total = len(images)
    train_end = int(0.7 * total)
    val_end = train_end + int(0.15 * total)

    splits = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:]
    }

    for split_name, file_list in splits.items():
        for file in file_list:
            src_path = os.path.join(source_folder, file)
            dst_path = os.path.join(base_dir, split_name, class_name, file)
            shutil.copy(src_path, dst_path)

    print(f"{class_name} split completed!")

split_and_copy(source_real, "real")
split_and_copy(source_fake, "fake")

print("All dataset splitting done!")
