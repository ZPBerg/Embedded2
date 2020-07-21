import mysql.connector
import datetime

from src.db.config import get_config
from contextlib import contextmanager, closing


class Table:
    def __init__(self):
        pass


class IMAGE(Table):
    """Image table with inputs as its columns in database.
    Class name must match exactly the spelling of the corresponding table in database

    To insert into database table using this class, use this class as input for function sql_insert:
        sql_insert(IMAGE('image_name', 'image_date', 'image_time'))

        Args:
            image_name (string) : name of image
            image_date (datetime obj) : date image was taken
            image_time (datetime obj) : time image was taken
    """

    def __init__(self, image_name: str, image_date: datetime, image_time: datetime):
        self.Image_Name = image_name
        self.Image_Date = image_date
        self.Image_Time = image_time


class BBOX(Table):
    """BBox table with inputs as its columns in database
    Class name must match exactly the spelling of the corresponding table in database

    To insert into database table using this class, use this class as input for function sql_insert:
        sql_insert(BBOX(xmin, ymin, xmax, ymax, conf, goggles, 'init_vector'))

        Args:

            xmin (float) : lower left x-coordinate of bounding box
            ymin (float) : lower left y-coordinate of bounding box
            xmax (float) : upper right x-coordinate of bouding box
            ymax (float) : upper right y-coordinate of bounding box
            conf (float) : confidence score for bounding box
            goggles (bool) : whether goggles were detected in bounding box
            image_name (string): name of image that contains the bounding box
            init_vector (string) : decryption key for encrypted bounding box
    """

    def __init__(self, xmin: int, ymin: int, xmax: int, ymax: int, conf: int, goggles: int, image_name: str, init_vector: str):
        self.X_Min = xmin
        self.Y_Min = ymin
        self.X_Max = xmax
        self.Y_Max = ymax
        self.Confidence = conf
        self.Goggles = goggles
        self.Image_Name = image_name
        self.Init_Vector = init_vector


@contextmanager
def sql_connection():
    """Sets up connection to mysql database

    Yields:
        sql connection: sql connector object to the sql database
    """
    conn_info = get_config()

    connection = mysql.connector.connect(
        host=conn_info["SQL_HOST"],
        user=conn_info["USER_NAME"],
        password=conn_info["PASSWORD"],
        database=conn_info["KEYSPACE"]
    )

    with closing(connection) as connection:
        yield connection


@contextmanager
def sql_cursor():
    """Gets sql cursor from database connection

    Yields:
       sql cursor : cursor object to database
    """
    with sql_connection() as connection:
        yield connection.cursor(buffered=True)
        connection.commit()


def sql_insert(table: Table):
    """Inserts row of information for a specified table in database

    Args:
        table (class obj): Class with name that corresponds to the table for data to be inserted into
    """
    key_list = ','.join([key for key, _ in table.__dict__.items()])
    value_list = ','.join([f'%({key})s' for key, _ in table.__dict__.items()])
    query = f"INSERT INTO {table.__class__.__name__}({key_list}) VALUES({value_list})"
    params = table.__dict__
    with sql_cursor() as cursor:
        try:
            cursor.execute(query, params)
        except Exception as e:
            print(e)


def sql_clear_table(table_name: Table):
    """Clears all rows in specified table in the database

    Args:
        table_name (string): name of table in database
    """
    query = f"DELETE FROM {table_name}"

    with sql_cursor() as cursor:
        try:
            cursor.execute(query)
        except Exception as e:
            print(e)
