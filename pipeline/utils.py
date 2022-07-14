import os
import glob


def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")


def make_feature_list(target_path, save_path, file_name):
    createDirectory(save_path)
    files = sorted(glob.glob(os.path.join(target_path, "*.npy")))
    with open(os.path.join(save_path, file_name), 'w+') as f:  ## the name of feature list
        for file in files:
            newline = file+'\n'
            f.write(newline)

    return os.path.join(save_path, file_name)

def linux_path_to_window(path):
    path = path.replace('\\', '/')

    return path