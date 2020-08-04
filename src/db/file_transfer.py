import paramiko
import os
import sys
from src.db.config import get_config
from contextlib import contextmanager


@contextmanager
def ftp_transfer():
    """Sets up connection to target machine using SFTP and returns transfer function

    Example Usage:
        with ftp_transfer() as transfer:
            transfer("./input_dir", "./output_dir", "file_name.ext")

    Yields:
        [function]: transfers file to output dir using SFTP

        Args:
            input_dir [string]: input directory for image, can be absolute or relative path
            output_dir [string]: output directory for image on target machine, can be absolute or relative path
            image_name [string]: name of file, include file type extension
    """
    transport = None
    sftp = None
    config = get_config()

    try:
        transport = paramiko.Transport((config["FTPHOST"], 22))
        transport.connect(
            username=config["FTPUSER"], password=config["FTPPASS"])
        sftp = paramiko.SFTPClient.from_transport(transport)
        print('Sucessfully connected to host machine')

        def transfer(input_dir, output_dir, image_name):
            # ensure correct format for directory paths
            if not input_dir.endswith('/'):
                input_dir = input_dir + '/'
            if not output_dir.endswith('/'):
                output_dir = output_dir + '/'

            # transfer files from input directory to output directory
            try:
                sftp.put(input_dir + image_name, output_dir + image_name)
                # remove the image from client side after successful transfer (commented for testing purpose)
                # os.remove(input_dir + image_name)

                return True
            except Exception as e:
                print(e)
                return False
        yield transfer

    except Exception as e:
        print(e)
        sys.exit()
    finally:
        sftp.close()
        transport.close()
