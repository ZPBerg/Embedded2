import argparse
import datetime
import json
import os
import subprocess

from src.db.db_connection import sql_cursor

"""
Put this file one folder up from the stored images.
Eg. on ee220clnx1: if /local/b/embedvis/Nano_Images contains images, 
this file's path should be /local/b/embedvis/collect_images.py
Set up a cron job to run this script daily.
rclone should be set up, in our case pointing to a Google Drive folder: https://rclone.org/drive/

Collect images of non-goggle detections from the database.
Upload images and metadata to Google Drive.
"""

METADATA_FILE = os.path.join(os.path.dirname(__file__), 'metadata.json')
TODAY = datetime.datetime.today().strftime('%Y-%m-%d')


def get_metadata():
    """
    Get image filenames and other relevant metadata from the database.
    @return: A list of dictionaries with the metadata for each image TODO describe the metadata

    Query:
    SELECT b.image_name, b.X_Min, b.Y_Min, b.X_Max, b.Y_Max,
    i.image_name, i.init_vector from bbox AS b, image as i where b.image_name=i.image_name and b.goggles=False
    """

    metadata = []
    #current_date = (datetime.date.today(),)

    # for testing
    date = datetime.datetime(2020, 7, 23)
    current_date = (date,)
    # for testing

    # make sql connection
    # execute query
    with sql_cursor() as cursor:
        try:
            cursor.execute('USE goggles')
            cursor.execute('SELECT b.image_name, b.X_Min, b.Y_Min, b.X_Max, b.Y_Max, '
                           'b.init_vector, b.goggles from BBOX AS b, IMAGE as i where '
                           'b.image_name=i.image_name and i.image_date=%s and b.goggles=False', current_date)

            for (image_name, x_min, y_min, x_max, y_max, init_vector, goggles) in cursor:
                metadata.append({'image_name': image_name,
                                 'x_min': float(x_min),  # JSON cannot serialize Decimals.
                                 'y_min': float(y_min),  # If there is a better way to do this, let me know.
                                 'x_max': float(x_max),
                                 'y_max': float(y_max),
                                 'init_vector': init_vector
                                 })
        except Exception as e:
            print(e)

    with open(METADATA_FILE, 'w') as meta_file:
        json.dump(metadata, meta_file)
    return metadata


def upload_files(metadata, dir, rclone_path, remote_name):
    """
    For each filename returned by get_metadata, upload image
    to Drive. Upload the day's metadata file.
    @param metadata: the list of dictionaries returned by get_metadata
    @param dir: the folder containing the images to upload
    @param rclone_path: path to rclone installation. Must be an absolute path if on the HELPS machine.
    @param remote_name: name of remote location in rclone
    """

    # prevent sending the same image twice (if two faces are detected)
    images = []

    # send images to the Drive
    for image in metadata:
        if image not in images:
            images.append(image)
            image_path = os.path.join(os.path.dirname(__file__), dir, image['image_name'])
            subprocess.run([rclone_path, 'copy', image_path, '{}:{}'.format(remote_name, TODAY)])

    # upload metadata json to the Drive
    subprocess.run([rclone_path, 'copy', METADATA_FILE, '{}:{}'.format(remote_name, TODAY)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Collect images.')
    parser.add_argument('--directory', '-d', type=str, required=True, help='Folder containing images to upload')
    parser.add_argument('--rclone_path', '-r', type=str, default='rclone', help='Path to rclone installation. If not '
                                                                                'on the HELPS machine, the default '
                                                                                'should work (if you have rclone '
                                                                                'installed).')
    parser.add_argument('--remote_name', type=str, default='EmbedVisDrive', help='Name of remote location according '
                                                                                  'to rclone (default is the Drive '
                                                                                  'name on the HELPS machine). Don\'t '
                                                                                  'include the semicolon.')
    args = parser.parse_args()

    # call the methods
    metadata = get_metadata()
    upload_files(metadata, args.directory, args.rclone_path, args.remote_name)

    exit(0)
