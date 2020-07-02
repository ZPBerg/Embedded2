import os
import argparse
import uuid

"""
Give unique names to all image files in a given folder (and its subfolders).
"""

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']


def generate_unique_name(include_index=False, index=None):
    """
    Generate a unique name with (for convenience in cleaning data for training)
    or without index before name.
    """

    unique_name = uuid.uuid4()

    if include_index:
        given_name = unique_name
    else:
        given_name = str(index) + '__' + str(unique_name)
        index += 1

    return given_name


def get_all_files(input_directory):
    """
    Return all files to be renamed (to a unique name) as a list of strings.
    """

    filenames = []

    for dir_name, subdir_list, file_list in os.walk(input_directory):
        print('Found directory: {}'.format(dir_name))

        for filename in file_list:
            ext = '.' + filename.split('.')[-1]
            if ext in IMG_EXTENSIONS:
                filenames.append(os.path.join(dir_name, filename))

    return filenames


def get_file_extension(filename):
    return filename.split('.')[-1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Name Changer")
    parser.add_argument('--input_directory', '-i', type=str, required=True, help="Input directory containing files")
    parser.add_argument('--output_directory', '-o', default='new_name_directory', action='store_true',
                        help="Output directory to save files")
    parser.add_argument('--include_index', default=False, action='store_true',
                        help="Prepend filenames with a number")
    args = parser.parse_args()

    filenames = get_all_files(args.input_directory)
    print('{} files found'.format(len(filenames)))

    index = 0

    for filename in filenames:
        new_name = generate_unique_name(args.include_index, index) + '.' + get_file_extension(filename)
        new_path = filename.replace(filename.split('/')[-1], new_name)

        # maintain folder structure, only renaming the file itself
        os.replace(filename, new_path)
        index += 1

    exit(0)
