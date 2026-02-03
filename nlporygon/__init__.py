from enum import Enum

from dotenv import load_dotenv


class SupportedDbType(str, Enum):
    DUCK_DB = "duck-db"


load_dotenv()
