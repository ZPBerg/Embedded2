import json

config = None
"""Cached configuation info

Config Specification:
    SQL_HOST : database domain
    USER_NAME : database user name
    PASSWORD : database password
    KEYSPACE : keyspace name
    FTPHOST : remote storage domain
    FTPUSER : remote storage user name
    FTPPASS : remote storage password
"""


def get_config():
    """Returns database connection information

    Returns:
        [dict]: database and remote storage connection info
    """
    global config

    if config:
        return config

    with open('login.json') as file:
        config = json.load(file)

    return config
