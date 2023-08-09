import os
from pathlib import Path

from PIL import Image

BASE_SNC_DIR: Path = Path.cwd().parent
DATA_DIR: Path = (BASE_SNC_DIR / 'data')
CROP_DIR: Path = (DATA_DIR / 'crops')
RESIZED_DIR: Path = (CROP_DIR / 'resized')
ORIGINAL_CROPS_DIR = CROP_DIR / 'original'


def main(argv=None):
    # Create the resized directory if it doesn't exist
    RESIZED_DIR.mkdir(parents=True, exist_ok=True)

    # List all files in the original directory
    files = ORIGINAL_CROPS_DIR.glob('*')

    # Loop through the files in the original directory
    for file_path in files:
        # Open the image using PIL
        img = Image.open(file_path)

        # Get the dimensions of the image
        width, height = img.size

        # Check if the image dimensions meet the criteria
        if width >= 30 and height >= 60:
            # Resize the image to 30x60 pixels
            img = img.resize((30, 60), Image.LANCZOS)
            # Construct the path for the resized image
            resized_path = RESIZED_DIR / file_path.name

            # Save the resized image to the resized directory
            img.save(resized_path)


if __name__ == '__main__':
    main()
