import datetime
import json

# from Embedded2.src.jetson.db ... import ...

"""
Put this file one folder up from the stored images.
Eg. /local/b/embedvis/imgs contains images, /local/b/embedvis/collect_images.py

Collect images of non-goggle detections from the database.
Upload images to Google Drive.
Email end-user with the Drive link.
"""


def query_db():
    """Get image filenames. Probably just a SQL query."""
    pass


def upload_images(imgs):
    """
    For each filename returned by query_db, upload image
    and its relevant metadata (eg. face coords) to Drive.
    @param imgs: [str, str, ...]
    """

    current_date = datetime.datetime.now().strftime("%m-%d-%Y")

    # TODO how should metadata be transferred? JSON file?
    with open(current_date + '.json', 'w') as meta_file:
        for i in imgs:
            # 1. append image metadata
            # 2. upload image
            image_metadata = []
            json.dump(image_metadata, meta_file)
            pass

        # upload metadata json file to Drive


if __name__ == "__main__":
    # call the methods
    pass
