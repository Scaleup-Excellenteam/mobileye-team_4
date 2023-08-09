# This file contains constants.. Mainly strings.
# It's never a good idea to have a string scattered in your code across different files, so just put them here
from pathlib import Path
from typing import List

# # Column name
SEQ_IMAG: str = 'seq_imag'  # Serial number of the image
NAME: str = 'name'
IMAG_PATH: str = 'imag_path'
GTIM_PATH: str = 'gtim_path'
JSON_PATH: str = 'json_path'
X: str = 'x'
Y: str = 'y'
COLOR: str = 'color'

# # Data CSV columns:
CSV_INPUT: List[str] = [SEQ_IMAG, NAME, IMAG_PATH, JSON_PATH, GTIM_PATH]
CSV_OUTPUT: List[str] = [SEQ_IMAG, NAME, IMAG_PATH, JSON_PATH, GTIM_PATH, X, Y, COLOR]

TRAIN_TEST_VAL = 'train_test_val'
TRAIN = 'train'
TEST = 'test'
VALIDATION = 'validation'

BASE_SNC_DIR = Path.cwd().parent
DATA_DIR: Path = (BASE_SNC_DIR / 'data')
FULL_IMAGES_DIR: Path = 'fullImages'  # Where we write the full images
CROP_DIR: Path = DATA_DIR / 'crops'

# # Crop size:
DEFAULT_CROPS_W: int = 32
DEFAULT_CROPS_H: int = 96

SEQ: str = 'seq'  # The image seq number -> for tracing back the original image
IS_TRUE: str = 'is_true'  # Is it a traffic light or not.
IS_IGNORE: str = 'is_ignore'
# investigate the reason after
CROP_PATH: str = 'path'
X0: str = 'x0'  # The bigger x value (the right corner)
X1: str = 'x1'  # The smaller x value (the left corner)
Y0: str = 'y0'  # The smaller y value (the lower corner)
Y1: str = 'y1'  # The bigger y value (the higher corner)
COL: str = 'col'

RELEVANT_IMAGE_PATH: str = 'path'
ZOOM: str = 'zoom'  # If you zoomed in the picture, then by how much? (0.5. 0.25 etc.).
PATH: str = 'path'

# # CNN input CSV columns:
CROP_RESULT: List[str] = [SEQ, IS_TRUE, IS_IGNORE, CROP_PATH, X0, X1, Y0, Y1, COL]
ATTENTION_RESULT: List[str] = [RELEVANT_IMAGE_PATH, X, Y, ZOOM, COL]

# # Files path
BASE_SNC_DIR: Path = Path.cwd().parent
ATTENTION_PATH: Path = DATA_DIR / 'attention_results'

ATTENTION_CSV_NAME: str = 'attention_results.csv'
CROP_CSV_NAME: str = 'crop_results.csv'

MODELS_DIR: Path = DATA_DIR / 'models'  # Where we explicitly copy/save good checkpoints for "release"
LOGS_DIR: Path = MODELS_DIR / 'logs'  # Each model will have a folder. TB will show all models


# # File names (directories to be appended automatically)
TFLS_CSV: str = 'tfls.csv'
CSV_OUTPUT_NAME: str = 'results.csv'