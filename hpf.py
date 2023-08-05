import os
from typing import List, Optional, Union, Dict, Tuple
import json
import argparse
from pathlib import Path
from matplotlib.colors import Normalize
from scipy.ndimage import maximum_filter
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from datetime import datetime

# if you wanna iterate over multiple files and json, the default source folder name is this.
DEFAULT_BASE_DIR: str = 'INSERT_YOUR_DIR_WITH_PNG_AND_JSON_HERE'

# The label we wanna look for in the polygons json file
TFL_LABEL = ['traffic light']

POLYGON_OBJECT = Dict[str, Union[str, List[int]]]
RED_X_COORDINATES = List[int]
RED_Y_COORDINATES = List[int]
GREEN_X_COORDINATES = List[int]
GREEN_Y_COORDINATES = List[int]

RED_LIGHT = 'RED'
GREEN_LIGHT = 'GREEN'


def high_pass_filter(image, kernel):
    filtered_image = np.zeros_like(image, dtype=float)
    for channel in range(image.shape[2]):
        filtered_image[:, :, channel] = convolve(image[:, :, channel].astype(np.float32), kernel)
    return filtered_image


def find_tfl_lights(c_image: np.ndarray, **kwargs) -> Tuple[
    RED_X_COORDINATES, RED_Y_COORDINATES, GREEN_X_COORDINATES, GREEN_Y_COORDINATES]:
    """
    Detect candidates for TFL lights.

    Args:
        c_image (np.ndarray): The image itself as np.uint8, shape of (H, W, 3).
        kwargs: Whatever config you want to pass in here.

    Returns:
        4-tuple of x_red, y_red, x_green, y_green.
    """
    # kernel = np.array(
    #     [
    #         [-1, -1, -1],
    #         [-1, 8, -1],
    #         [-1, -1, -1]
    #     ]
    # )
    # kernel = np.array(
    #     [
    #         [-1, -1, -1, -1, -1],
    #         [-1, -1, 4, -1, -1],
    #         [-1, 4, 5, 4, -1],
    #         [-1, -1, 4, -1, -1],
    #         [-1, -1, -1, -1, -1]
    #     ]
    # )

    kernel = np.array(
        [
            [0, -0.25, 0],
            [-0.25, 2, -0.25],
            [0, -0.25, 0]
        ]
    )
    # Apply high-pass filter
    filtered_image = high_pass_filter(c_image, kernel)
    norm = Normalize()
    filtered_image = norm(filtered_image)
    plt.imshow(filtered_image)

    return [], [], [], []


def show_image_and_gt(image, objs, fig_num=None):
    plt.figure(fig_num)
    plt.imshow(image)
    labels = set()
    if objs is not None:
        for obj in objs:
            if obj['label'] == 'traffic light':
                polygon = np.array(obj['polygon']).reshape((-1, 1, 2))
                plt.plot(polygon[:, :, 0], polygon[:, :, 1], 'g')
                labels.add(obj['label'])

        print(f"Figure {fig_num}: {labels}")


def test_find_tfl_lights(image_path, json_path=None, fig_num=None):
    """
    Run the attention code.
    """
    image = np.array(Image.open(image_path))
    if json_path is None:
        objects = None
    else:
        json_data = json.load(open(json_path))
        objects = json_data['objects']
    show_image_and_gt(image, objects, fig_num)

    red_x, red_y, green_x, green_y = find_tfl_lights(image, some_threshold=42)
    plt.plot(red_x, red_y, 'ro', color='r', markersize=4)
    plt.plot(green_x, green_y, 'ro', color='g', markersize=4)


def main(argv=None):
    """
    It's nice to have a standalone tester for the algorithm.
    Consider looping over some images from here, so you can manually examine the results.
    Keep this functionality even after you have all system running, because you sometime want to debug/improve a module.

    :param argv: In case you want to programmatically run this.
    """
    parser = argparse.ArgumentParser("Test TFL attention mechanism")
    parser.add_argument('-i', '--image', type=str, help='Path to an image')
    parser.add_argument("-j", "--json", type=str, help="Path to image json file -> GT for comparison")
    parser.add_argument('-d', '--dir', type=str, help='Directory to scan images in')
    args = parser.parse_args(argv)

    # If you entered a custom dir to run from or the default dir exist in your project then:
    directory_path: Path = Path(args.dir or DEFAULT_BASE_DIR)
    if directory_path.exists():
        # gets a list of all the files in the directory that ends with "_leftImg8bit.png".
        file_list: List[Path] = list(directory_path.glob('*_leftImg8bit.png'))

        for image in file_list:
            # Convert the Path object to a string using as_posix() method
            image_path: str = image.as_posix()
            path: Optional[str] = image_path.replace('_leftImg8bit.png', '_gtFine_polygons.json')
            image_json_path: Optional[str] = path if Path(path).exists() else None
            test_find_tfl_lights(image_path, image_json_path)

    if args.image and args.json:
        test_find_tfl_lights(args.image, args.json)
    elif args.image:
        test_find_tfl_lights(args.image)
    plt.show(block=True)


if __name__ == '__main__':
    main()
