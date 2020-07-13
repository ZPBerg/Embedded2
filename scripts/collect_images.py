import argparse
import datetime
import json

from src.jetson.db.db_connection import sql_cursor

# from wherever import email method

"""
Put this file one folder up from the stored images.
Eg. on the HELPS machine: if /local/b/embedvis/imgs contains images, 
this file's path should be /local/b/embedvis/collect_images.py

Collect images of non-goggle detections from the database.
Upload images and metadata to Google Drive.
Email end-user with the Drive link.
"""

METADATA_FILE = 'metadata.json'


# TODO rename method
def get_metadata():
    """
    Get image filenames and other relevant metadata from the database.
    @return: A list of dictionaries with the metadata for each image TODO describe the metadata

    Query:
    SELECT b.image_name, b.X_Min, b.Y_Min, b.X_Max, b.Y_Max,
    i.image_name, i.init_vector from bbox AS b, image as i where b.image_name=i.image_name and b.goggles=False
    """

    metadata = []

    # make sql connection
    # execute query
    with sql_cursor() as cursor:
        try:
            cursor.execute('USE goggles')
            cursor.execute('SELECT b.image_name, b.X_Min, b.Y_Min, b.X_Max, b.Y_Max, '
                           'i.image_name, i.init_vector from bbox AS b, image as i where '
                           'b.image_name=i.image_name and b.goggles=False')

            for (image_name, x_min, y_min, x_max, y_max, image_name, init_vector) in cursor:
                metadata.append({'image_name': image_name,
                                 'x_min': float(x_min),  # JSON cannot serialize Decimals.
                                 'y_min': float(y_min),  # If there is a better way to do this, someone let me know.
                                 'x_max': float(x_max),
                                 'y_max': float(y_max),
                                 'init_vector': init_vector
                                 })
        except Exception as e:
            print(e)

    with open(METADATA_FILE, 'w') as meta_file:
        json.dump(metadata, meta_file)
    return metadata


# TODO make folder with date to contain images and metadata file
def upload_files(metadata, dir):
    """
    For each filename returned by get_metadata, upload image
    to Drive. Upload the day's metadata file.
    @param metadata: the list of dictionaries returned by get_metadata
    @param dir: the folder containing the images to upload
    """

    for image in metadata:
        # upload image using rclone
        # subprocess rclone copy os.path.join(dir, image['image_name']) [Drive name]
        pass

    # upload metadata json file to Drive
    # subprocess rclone copy METADATA_FILE [Drive name]:


# TODO call Seoyoung's method to email

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Collect images.')
    parser.add_argument('--directory', '-d', type=str, required=True, help='Folder containing images to upload')
    args = parser.parse_args()

    current_date = datetime.datetime.now().strftime("%m-%d-%Y")

    # call the methods
    metadata = get_metadata()
    upload_files(metadata, args.directory)
