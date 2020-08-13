import argparse
import datetime
import json
import os
import subprocess

from src.db.db_connection import sql_cursor

"""
This script should be set up with a cron job to run daily.
Each user will have to set up their rclone config, in our case pointing to a Google Drive folder: https://rclone.org/drive/
This can be done with /local/b/embedvis/rclone-v1.52.2-linux-amd64/rclone config

Collect images of non-goggle detections from the database.
Upload images and metadata to Google Drive.
"""

METADATA_FILE = 'metadata.json'


def get_metadata():
    """
    Get image filenames and other relevant metadata from the database.
    Save metadata to a file for future decryption.
    @return: A list of dictionaries with the metadata for each image

    Example list: [
    {'image_name': "0.jpg", 'x_min': 0.0, 'y_min': 0.0, 'x_max': 100.0, 'y_max': 100.0, 'init_vector': "example"}
    {'image_name': "1.jpg", 'x_min': 25.0, 'y_min': 25.0, 'x_max': 120.0, 'y_max': 140.0, 'init_vector': "example2"}]
    """

    metadata = []
    current_date = (datetime.date.today(),)

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
                                 'x_min': float(x_min),
                                 'y_min': float(y_min),
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
    @param remote_name: name of remote location in rclone
    """

    images = []
    today = datetime.datetime.today().strftime('%Y-%m-%d')

    # send images to the Drive
    for image in metadata:
        # prevent sending the same image twice (if two faces are detected)
        if image not in images:
            images.append(image)
            image_path = os.path.join(dir, image['image_name'])
            subprocess.run([rclone_path, 'copy', image_path, '{}:{}'.format(remote_name, today)])

    # upload metadata.json to the Drive
    subprocess.run([rclone_path, 'copy', METADATA_FILE, '{}:{}'.format(remote_name, today)])
    os.remove(METADATA_FILE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Collect images.')
    parser.add_argument('--directory', '-d', type=str, required=True, help='Folder containing images to upload')
    parser.add_argument('--rclone_path', '-r', type=str, default='/local/b/embedvis/rclone-v1.52.2-linux-amd64/rclone',
                        help='Location of rclone binary. Default version on ee220clnx1 doesn\'t support copying to '
                             'shared folders.')
    parser.add_argument('--remote_name', type=str, help='Name of remote location according to rclone config. You must '
                                                        'create your own config.')
    args = parser.parse_args()

    metadata = get_metadata()
    upload_files(metadata, args.directory, args.rclone_path, args.remote_name)

    exit(0)
