import os
from typing import List, Optional, Union, Dict, Tuple, Any
import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.ndimage import maximum_filter
from scipy.signal import convolve
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from sklearn.cluster import DBSCAN
from datetime import datetime

# if you wanna iterate over multiple files and json, the default source folder name is this.
DEFAULT_BASE_DIR: str = 'resources'

# The label we wanna look for in the polygons json file
TFL_LABEL = ['traffic light']

POLYGON_OBJECT = Dict[str, Union[str, List[int]]]
RED_X_COORDINATES = List[int]
RED_Y_COORDINATES = List[int]
GREEN_X_COORDINATES = List[int]
GREEN_Y_COORDINATES = List[int]

SEQ: str = 'seq'  # The image seq number -> for tracing back the original image
IS_TRUE: str = 'is_true'  # Is it a traffic light or not.
IGNOR: str = 'is_ignore'  # If it's an unusual crop (like two tfl's or only half etc.) that you can just ignore it and
# investigate the reason after
CROP_PATH: str = 'path'
X0: str = 'x0'  # The bigger x value (the right corner)
X1: str = 'x1'  # The smaller x value (the left corner)
Y0: str = 'y0'  # The smaller y value (the lower corner)
Y1: str = 'y1'  # The bigger y value (the higher corner)
COL: str = 'col'
SEQ_IMAG: str = 'seq_imag'  # Serial number of the image
GTIM_PATH: str = 'gtim_path'
X: str = 'x'
Y: str = 'y'
COLOR: str = 'color'
CROP_RESULT: List[str] = [SEQ, IS_TRUE, IGNOR, CROP_PATH, X0, X1, Y0, Y1, COL]

# Files path
BASE_SNC_DIR: Path = Path.cwd().parent
DATA_DIR: Path = (BASE_SNC_DIR / 'data')
CROP_DIR: Path = DATA_DIR / 'crops'
ATTENTION_PATH: Path = DATA_DIR / 'attention_results'

CROP_CSV_NAME: str = 'crop_results.csv'  # result CSV name

RED_LIGHT = 'red'
GREEN_LIGHT = 'green'
TIMESTAMP_FORMAT = "%Y%m%d%H%M%S"


def create_color_masks(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create color masks for red and green colors in the image.

    Args:
        image (np.ndarray): The input image as a NumPy array.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The red and green masks.
    """
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Define red and green traffic lights RGB values
    red_lights_rgb = [(255, 98, 29), (157, 63, 45), (255, 211, 106), (255, 195, 164), (229, 119, 118)]
    green_lights_rgb = [(0, 255, 255), (0, 204, 204), (144, 238, 144), (0, 173, 220), (88, 218, 208)]

    # Convert the RGB values to HSV
    red_lights_hsv = [cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2HSV)[0][0] for rgb in red_lights_rgb]
    green_lights_hsv = [cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2HSV)[0][0] for rgb in green_lights_rgb]

    # Define lower and upper bounds for red and green colors
    lower_red, upper_red = np.min(red_lights_hsv, axis=0), np.max(red_lights_hsv, axis=0)
    lower_green, upper_green = np.min(green_lights_hsv, axis=0), np.max(green_lights_hsv, axis=0)

    # Create masks for red and green colors
    red_mask = cv2.inRange(hsv_image, lower_red, upper_red)
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

    return red_mask, green_mask


def high_pass_filter(image: np.ndarray, kernel_size: Tuple[int, int] = (11, 11), sigma: float = 3.0) -> np.ndarray:
    """
    Apply a Gaussian high-pass filter to the input image to enhance edges and details.
    Args:
        image (np.ndarray): The input image as a NumPy array.
        kernel_size (Tuple[int, int]): The size of the Gaussian kernel. The larger the kernel, the more the image will be blurred.
        sigma (float): The standard deviation for Gaussian kernel. The higher sigma, the more the image will be blurred.
    Returns:
        np.ndarray: The filtered image.
    """

    # Apply Gaussian blur to the image
    low_pass = cv2.GaussianBlur(image, kernel_size, sigma)

    # Subtract the low-pass filtered image from the original image and add 20 offset
    high_pass = image - low_pass + 20

    # Ensure that the values are in the valid range 0-255
    high_pass = np.clip(high_pass, 0, 255)

    return high_pass.astype(np.float64)


def filter_edge_points(x_coords, y_coords, width, height, diameters, margin=5):
    """
    Filter out the points near the edges of the image.

    Args:
        x_coords, y_coords: The x and y coordinates.
        width, height: The dimensions of the image.
        margin: The margin size to consider as an edge.

    Returns:
        filtered_x, filtered_y: The filtered x and y coordinates.
    """
    filtered_x = []
    filtered_y = []
    filtered_diameters = []
    for x, y, diameter in zip(x_coords, y_coords, diameters):
        if margin < x < width - margin and margin < y < height - margin:
            filtered_x.append(x)
            filtered_y.append(y)
            filtered_diameters.append(diameter)
    return filtered_x, filtered_y, filtered_diameters


# TFL detection function
def apply_filters_and_threshold(image_mask, percentile=99.9):
    """
    Apply high-pass filters and threshold the convolved image.

    Args:
        image_mask (np.ndarray): The input color mask.
        percentile (float): Percentile for thresholding.

    Returns:
        np.ndarray: The thresholded and filtered image.
    """
    kernel = np.array([
        [-1, -1, -1, -1, -1],
        [-1, 1, 2, 1, -1],
        [-1, 2, 4, 2, -1],
        [-1, 1, 2, 1, -1],
        [-1, -1, -1, -1, -1]
    ])
    edges = high_pass_filter(image_mask)
    convolved = convolve(edges.astype(np.float64), kernel, "same")
    threshold = convolved > np.percentile(convolved, percentile)
    filtered = maximum_filter(threshold, 5)
    return filtered


def cluster_and_find_centroids(x_coords, y_coords, eps=5, min_samples=2):
    """
    Cluster the given coordinates and find the centroids.

    Args:
        x_coords, y_coords: The x and y coordinates.
        eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        min_samples: The number of samples in a neighborhood for a point to be considered as a core point.

    Returns:
        centroids_x, centroids_y: The x and y coordinates of the centroids.
    """
    coordinates = np.column_stack((x_coords, y_coords))
    if len(coordinates) == 0:
        return [], []

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coordinates)
    labels = clustering.labels_

    centroids_x = []
    centroids_y = []
    diameters = []

    for cluster_label in set(labels):
        if cluster_label == -1:  # Ignore noise
            continue

        cluster_points = coordinates[labels == cluster_label]
        centroid_x = np.mean(cluster_points[:, 0])
        centroid_y = np.mean(cluster_points[:, 1])
        centroids_x.append(centroid_x)
        centroids_y.append(centroid_y)

        if len(cluster_points) > 1:
            pairwise_distances = pdist(cluster_points)
            diameter = np.max(pairwise_distances)
            diameters.append(diameter)

    return centroids_x, centroids_y, diameters  # , farthest_averages_x, farthest_averages_y


def find_tfl_coordinates(color_mask):
    """
    Find traffic light coordinates for a specific color mask.
    Args:
        color_mask (np.ndarray): The color mask.
    Returns:
        x, y: The x and y coordinates of the detected traffic lights.
    """
    filtered_mask = apply_filters_and_threshold(color_mask)
    y, x = np.nonzero(filtered_mask)
    diameters = []
    centroids_x, centroids_y, diameters = cluster_and_find_centroids(x, y)
    return centroids_x, centroids_y, diameters


def find_tfl_lights(c_image: np.ndarray, **kwargs):
    # -> Tuple[
    # RED_X_COORDINATES, RED_Y_COORDINATES, GREEN_X_COORDINATES, GREEN_Y_COORDINATES]:
    """
    Find traffic light coordinates in the given image.

    Args:
        c_image (np.ndarray): The input image as a NumPy array.

    Returns:
        Tuple: The x and y coordinates of the detected red and green traffic lights.
    """
    red_mask, green_mask = create_color_masks(c_image)
    red_x, red_y, red_diameters = find_tfl_coordinates(red_mask)
    green_x, green_y, green_diameters = find_tfl_coordinates(green_mask)

    red_x, red_y, red_diameters = filter_edge_points(red_x, red_y, c_image.shape[1], c_image.shape[0], red_diameters)
    green_x, green_y, green_diameters = filter_edge_points(green_x, green_y, c_image.shape[1], c_image.shape[0],
                                                           green_diameters)

    return list(red_x), list(red_y), list(green_x), list(green_y), list(red_diameters), list(green_diameters)


def show_image_and_gt(c_image: np.ndarray, objects: Optional[List[POLYGON_OBJECT]], fig_num: int = None):
    # ensure a fresh canvas for plotting the image and objects.
    plt.figure(fig_num).clf()
    # displays the input image.
    plt.imshow(c_image)
    labels = set()
    if objects:
        for image_object in objects:
            # Extract the 'polygon' array from the image object
            poly: np.array = np.array(image_object['polygon'])
            # Use advanced indexing to create a closed polygon array
            # The modulo operation ensures that the array is indexed circularly, closing the polygon
            polygon_array = poly[np.arange(len(poly)) % len(poly)]
            # gets the x coordinates (first column -> 0) anf y coordinates (second column -> 1)
            x_coordinates, y_coordinates = polygon_array[:, 0], polygon_array[:, 1]
            color = 'r'
            plt.plot(x_coordinates, y_coordinates, color, label=image_object['label'])
            labels.add(image_object['label'])
        if 1 < len(labels):
            # The legend provides a visual representation of the labels associated with the plotted objects.
            # It helps in distinguishing different objects in the plot based on their labels.
            plt.legend()


def is_coord_in_boundary(coord: Tuple[int, int], image: np.ndarray) -> bool:
    """
    Check if the given coordinate lies within the boundaries of the image.

    Args:
        coord (Tuple[int, int]): The (x, y) coordinate to check.
        image (np.ndarray): The image to check the coordinate against.

    Returns:
        bool: True if the coordinate is within the image boundaries, False otherwise.
    """
    return coord[0] > 0 and coord[0] < image.shape[1] and coord[1] > 0 and coord[1] < image.shape[0]


def draw_traffic_light_rectangles(image: np.ndarray, red_x: List[int], red_y: List[int], green_x: List[int],
                                  green_y: List[int],
                                  red_diameters: List[int], green_diameters: List[int]) \
        -> Tuple[
            np.ndarray, List[Tuple[Tuple[int, int], Tuple[int, int]]], List[Tuple[Tuple[int, int], Tuple[int, int]]]]:
    """
    Illustrates rectangles around identified traffic lights in an image.

    This function draws bounding rectangles for both red and green traffic lights based on provided coordinates and
    diameters. It returns the modified image along with the coordinates of the rectangles.

    Args:
        image (np.ndarray): The source image.
        red_x, red_y: X and Y coordinates for detected red traffic lights.
        green_x, green_y: X and Y coordinates for detected green traffic lights.
        red_diameters: Diameters of the detected red traffic lights.
        green_diameters: Diameters of the detected green traffic lights.

    Returns:
        np.ndarray: The modified image with traffic light bounding rectangles.
        List[Tuple]: List of top-left and bottom-right coordinates of the rectangles for red traffic lights.
        List[Tuple]: List of top-left and bottom-right coordinates of the rectangles for green traffic lights.
    """

    red_rectangles = []
    green_rectangles = []

    # Make a copy of the image to avoid modifying the original
    image_with_rectangles = image.copy()

    # Draw rectangles for red lights (light at the top of the rectangle)
    for x, y, diameter in zip(red_x, red_y, red_diameters):
        top_left = (int(x - diameter / 2 - 5), int(y - diameter / 2 - 5))
        bottom_right = (int(x + diameter / 2 + 5), int(y + 3 * diameter))
        if not is_coord_in_boundary(top_left, image) or not is_coord_in_boundary(bottom_right, image):
            continue
        cv2.rectangle(image_with_rectangles, top_left, bottom_right, (255, 55, 0), 2)
        red_rectangles.append((top_left, bottom_right))

    # Draw rectangles for green lights (light at the bottom of the rectangle)
    for x, y, diameter in zip(green_x, green_y, green_diameters):
        top_left = (int(x - diameter / 2 - 5), int(y - 3 * diameter))
        bottom_right = (int(x + diameter / 2 + 5), int(y + diameter / 2 + 5))
        if not is_coord_in_boundary(top_left, image) or not is_coord_in_boundary(bottom_right, image):
            continue
        cv2.rectangle(image_with_rectangles, top_left, bottom_right, (0, 181, 26), 2)
        green_rectangles.append((top_left, bottom_right))

    return image_with_rectangles, red_rectangles, green_rectangles


def save_image(image: np.ndarray, output_path: str):
    """
    Save the given image to the specified path.

    Args:
        image (np.ndarray): The image to save.
        output_path (str): The path where the image will be saved.
    """
    # Check if the image is not empty
    if image.size > 0:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Save the image
        cv2.imwrite(str(output_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def check_crop(image_json_path: str, x0: int, y0: int, x1: int, y1: int) -> Tuple[bool, bool]:
    """
    Check if the crop contains a traffic light based on the ground truth.

    Args:
        image_json_path (str): The path to the image's JSON file.
        x0, y0, x1, y1: The coordinates of the crop.
        color (str): The color of the traffic light.

    Returns:
        Tuple[bool, bool]: Whether the crop contains a traffic light and whether it should be ignored.
    """
    # Load the image's JSON file
    image_json = json.load(Path(image_json_path).open())

    # if the crop contains more than one traffic-light then ignore it.
    traffic_lights = sum(1 for image_object in image_json['objects'] if image_object['label'] == 'traffic light')
    if traffic_lights != 1:
        return False, True

    # Iterate over the objects in the JSON file
    for image_object in image_json['objects']:
        if image_object['label'] == 'traffic light':
            polygon_size = len(image_object['polygon'])
            # Check if the traffic light's coordinates are within the crop's coordinates
            contained_coords = sum(1 for x, y in image_object['polygon'] if x0 <= x <= x1 and y0 <= y <= y1)
            # if half or more of the polygon is contained in the crop it's a traffic-light
            if contained_coords >= 0.5 * polygon_size:
                # The crop contains a traffic-light
                return True, False

    # The crop does not contain a traffic-light
    return False, True


def save_for_part_2(crops_df: pd.DataFrame):
    """
    *** No need to touch this. ***
    Saves the result DataFrame containing the crops data in the relevant folder under the relevant name for part 2.
    """
    if not ATTENTION_PATH.exists():
        ATTENTION_PATH.mkdir()
    crops_sorted: pd.DataFrame = crops_df.sort_values(by=SEQ)
    crops_sorted.to_csv(ATTENTION_PATH / CROP_CSV_NAME, index=False)


def create_crops(image: np.ndarray,
                 red_rectangles: List[Tuple[Tuple[float, float], Tuple[float, float]]],
                 green_rectangles: List[Tuple[Tuple[float, float], Tuple[float, float]]],
                 image_json_path: Optional[str] = None) -> pd.DataFrame:
    """
    Create crops around the detected traffic lights in the image.

    Args:
        image (np.ndarray): The input image as a NumPy array.
        red_rectangles: List of tuples with the top-left and bottom-right coordinates of the red traffic lights.
        green_rectangles: List of tuples with the top-left and bottom-right coordinates of the green traffic lights.
        image_json_path (str): The path to the image's JSON file.

    Returns:
        DataFrame: The DataFrame containing the information about the crops.
    """
    # Initialize the result DataFrame
    result_df = pd.DataFrame(columns=CROP_RESULT)

    # Initialize the template for the result
    result_template: Dict[str, Any] = {SEQ: '', IS_TRUE: '', IGNOR: '', CROP_PATH: '', X0: '', X1: '', Y0: '', Y1: '',
                                       COL: ''}

    # Combine colors and rectangles
    colors = [RED_LIGHT] * len(red_rectangles) + [GREEN_LIGHT] * len(green_rectangles)
    rectangles = red_rectangles + green_rectangles

    for i, (color, (top_left, bottom_right)) in enumerate(zip(colors, rectangles)):
        result_template[SEQ] = i
        result_template[COL] = color

        # Crop the image using the provided coordinates
        cropped_image = image[int(top_left[1]):int(bottom_right[1]), int(top_left[0]):int(bottom_right[0])]
        result_template[X0], result_template[Y0] = top_left
        result_template[X1], result_template[Y1] = bottom_right

        timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)
        crop_path = f'traffic_lights/{color}_light_{i}_{timestamp}.png'
        save_image(cropped_image, (CROP_DIR / crop_path).as_posix())
        result_template[CROP_PATH] = crop_path

        if image_json_path:
            result_template[IS_TRUE], result_template[IGNOR] = check_crop(image_json_path, *top_left, *bottom_right)
        else:
            result_template[IS_TRUE], result_template[IGNOR] = False, True

        result_df = result_df._append(result_template, ignore_index=True)

    return result_df


def test_find_tfl_lights(image_path: str, image_json_path: Optional[str] = None, fig_num=None):
    """
    Run the traffic light detection code.
    """
    # Using pillow to load the image
    image: Image = Image.open(image_path)
    # Converting the image to a numpy ndarray array
    c_image: np.ndarray = np.array(image)

    objects = None
    if image_json_path:
        image_json = json.load(Path(image_json_path).open())
        objects: List[POLYGON_OBJECT] = [image_object for image_object in image_json['objects']
                                         if image_object['label'] in TFL_LABEL]

    show_image_and_gt(c_image, objects, fig_num)

    red_x, red_y, green_x, green_y, red_diameters, green_diameters = find_tfl_lights(c_image)

    # Draw rectangles around the detected traffic lights
    c_image_with_rectangles, red_rectangles, green_rectangles = draw_traffic_light_rectangles(c_image, red_x, red_y,
                                                                                              green_x, green_y,
                                                                                              red_diameters,
                                                                                              green_diameters)
    # Display the image with rectangles
    plt.imshow(c_image_with_rectangles)

    # 'ro': This specifies the format string. 'r' represents the color red, and 'o' represents circles as markers.
    plt.plot(red_x, red_y, 'ro', markersize=4)
    plt.plot(green_x, green_y, 'go', markersize=4)

    # Create crops and add them to the DataFrame
    df = create_crops(c_image, red_rectangles, green_rectangles, image_json_path)

    # Save the DataFrame for part 2
    save_for_part_2(df)


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
