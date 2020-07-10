import argparse
import json
import os

"""
After having run collect_images, decrypt the associated images
(if necessary) and combine images together into a short video (using metadata).
"""

METADATA_FILE = 'metadata.json'


def decrypt_images(dir):
    # ask for decryption key
    with open(os.path.join(dir, METADATA_FILE)) as meta_file:
        metadata = json.load(meta_file)
        # use face coords to find where to decrypt in video frame
        # decrypt
        pass


def make_videos(dir):
    # use a heuristic (such as images within 5 seconds of each other)
    # to combine similar images into one video for easier viewing
    with open(os.path.join(dir, METADATA_FILE)) as meta_file:
        metadata = json.load(meta_file)
        # for each image, if within 5 seconds of the previous one,
        # concatenate them and make them into a video
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Combine images into a short video for easier viewing.'
                                     'Decrypt if needed.')
    parser.add_argument('--directory', '-d', type=str, required=True, help='Folder of images to be prepared.')
    parser.add_argument('--decrypt', default=False, action='store_true', help='Decrypt faces in the images.')
    parser.add_argument('--make_videos', '-m', default=False, action='store_true',
                        help='Combine frames from the same time period into a single video.')

    args = parser.parse_args()

    if not args.decrypt and not args.make_videos:
        print('No options selected. Please select at least one of --decrypt or --make_videos.')
        exit(0)

    if args.decrypt:
        decrypt_images(args.directory)

    if args.make_videos():
        make_videos(args.directory)
