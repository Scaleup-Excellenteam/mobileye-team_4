from typing import List, Optional, Union, Dict, Tuple
import json
import argparse
from pathlib import Path
from scipy.ndimage import maximum_filter
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

# if you wanna iterate over multiple files and json, the default source folder name is this.
DEFAULT_BASE_DIR: str = 'INSERT_YOUR_DIR_WITH_PNG_AND_JSON_HERE'

# The label we wanna look for in the polygons json file
TFL_LABEL = ['traffic light']

POLYGON_OBJECT = Dict[str, Union[str, List[int]]]
RED_X_COORDINATES = List[int]
RED_Y_COORDINATES = List[int]
GREEN_X_COORDINATES = List[int]
GREEN_Y_COORDINATES = List[int]


def display_pictures(c_image: np.ndarray, preprocessed_image: np.ndarray):
    # Display the original and preprocessed images side by side
    plt.figure(figsize=(10, 5))

    # Plot the original image
    plt.subplot(1, 2, 1)
    plt.imshow(c_image)
    plt.title('Original Image')
    plt.axis('off')

    # Plot the preprocessed image
    plt.subplot(1, 2, 2)
    plt.imshow(preprocessed_image)
    plt.title('Preprocessed Image')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def normalize_image(image: np.ndarray) -> np.ndarray:
    # Normalize the image pixel values to [0, 1]
    min_pixel_value = np.min(image)
    max_pixel_value = np.max(image)
    normalized_image = (image - min_pixel_value) / (max_pixel_value - min_pixel_value)
    return normalized_image


def high_pass_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Apply a high-pass filter to the input image to enhance edges and details.

    Args:
        image (np.ndarray): The input image as a NumPy array.
        kernel_size (int): The size of the filter kernel. (default: 3)

    Returns:
        np.ndarray: The filtered image.
    """
    # Define a kernel for high-pass filtering
    kernel = np.array(
        [
            [-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1]
        ]
    )
    # Initialize an empty array to store the filtered image
    filtered_image = np.zeros_like(image)

    # Apply the high-pass filter to each channel of the image
    for channel in range(image.shape[2]):
        filtered_image[:, :, channel] = convolve(image[:, :, channel].astype(np.float32), kernel)

    # Clip the values to ensure they are in the valid range [0, 255]
    filtered_image = np.clip(filtered_image, 0, 255)

    # Exchange the dark and bright parts by subtracting the filtered image from the original image
    inverted_image = image.astype(np.int16) - filtered_image.astype(np.int16)
    inverted_image = np.clip(inverted_image, 0, 255)

    return inverted_image.astype(np.uint8)


def find_traffic_lights_by_color(c_image: np.ndarray, lower_color: np.ndarray, upper_color: np.ndarray) -> Tuple[
    List[int], List[int]]:
    """
    Detect traffic light candidates of a specific color in the image.

    Args:
        c_image (np.ndarray): The input image.
        lower_color (np.ndarray): The lower bound of the color range.
        upper_color (np.ndarray): The upper bound of the color range.

    Returns:
        Tuple[List[int], List[int]]: The x and y coordinates of the detected lights.
    """
    # Create a mask for the specified color
    mask = cv2.inRange(c_image, lower_color, upper_color)

    mask = maximum_filter(mask, size=20)
    # Find contours in the masked image
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

    # Initialize lists to hold contour coordinates
    x, y = [], []

    # Append coordinates of each light to the lists
    for contour in contours:
        (x_coord, y_coord), radius = cv2.minEnclosingCircle(contour)
        x.append(int(x_coord))
        y.append(int(y_coord))

    return x, y


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
    # Apply high-pass filter
    c_image = high_pass_filter(c_image)
    # Define color ranges for red and green
    lower_red = np.array([220, 0, 0])
    upper_red = np.array([255, 55, 55])
    lower_green = np.array([0, 200, 0])
    upper_green = np.array([100, 255, 100])

    # Find red and green traffic light candidates
    red_x, red_y = find_traffic_lights_by_color(c_image, lower_red, upper_red)
    green_x, green_y = find_traffic_lights_by_color(c_image, lower_green, upper_green)

    return red_x, red_y, green_x, green_y


### GIVEN CODE TO TEST YOUR IMPLENTATION AND PLOT THE PICTURES
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
