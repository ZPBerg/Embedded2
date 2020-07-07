import datetime
import json

# from Embedded2.src.jetson.db ... import ...
# from wherever import email method

"""
Put this file one folder up from the stored images.
Eg. /local/b/embedvis/imgs contains images, /local/b/embedvis/collect_images.py

Collect images of non-goggle detections from the database.
Upload images to Google Drive.
Email end-user with the Drive link.
"""


# TODO rename method
def get_metadata():
    """
    Get image filenames and other relevant metadata from the database.
    @return: A list of dictionaries with the metadata for each image TODO describe the metadata

    Query:
    SELECT b.image_name, b.X_Min, b.Y_Min, b.X_Max, b.Y_Max,
    i.image_name, i.init_vector from bbox AS b, image as i where b.image_name == i.image_name and b.goggles == False
    """
    # make sql connection
    # execute query

    # for everything returned:
            # combine everything into a dictionary
            # append dictionary to list

    # return list of dictionaries
    # TODO just json.dump entire list?
    return []

# TODO don't need this method if json.dump ing all dictionaries at once
def organize_metadata(metadata):
    """
    Create metadata file needed for decrypting images.
    @param metadata: the list of dictionaries returned by get_metadata
    """

    with open(meta_file, 'w') as m:
        for x in metadata:
            # append image metadata

            # use metadata param
            image_metadata = []
            json.dump(image_metadata, m)


def upload_files(metadata):
    """
    For each filename returned by get_metadata, upload image
    to Drive. Upload the day's metadata file.
    @param metadata: the list of dictionaries returned by get_metadata
    """

    for image in metadata:
        # upload image using rclone
        pass

    # upload metadata json file to Drive
    # subprocess rclone copy meta_file [name of Drive in rclone]:


# TODO call Seoyoung's method to email

if __name__ == "__main__":
    current_date = datetime.datetime.now().strftime("%m-%d-%Y")
    meta_file = current_date + '.json'

    # call the methods
    metadata = get_metadata()
    organize_metadata(metadata)
    upload_files(metadata)
