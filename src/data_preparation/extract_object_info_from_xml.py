import os
from shutil import move
from glob import glob
import pandas as pd
from xml.etree import ElementTree as et
from functools import reduce
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_processing.log"),
        logging.StreamHandler()
    ]
)

xml_list = glob("data_images/*.xml")

def extract_text(filename):
    """
    Extracts relevant information from an XML file for YOLO model training.

    This function parses the XML file, extracts the image filename, its dimensions,
    and the bounding box coordinates for each labeled object within the image.

    Parameters
    ----------
    filename : str
        The path to the XML file to be processed.

    Returns
    -------
    list of lists
        A list where each sublist contains:
        [filename, width, height, name, xmin, xmax, ymin, ymax]

        Returns an empty list if an error occurs during processing.
    """
    try:
        tree = et.parse(filename)
        root = tree.getroot()
    except et.ParseError as e:
        logging.error(f"Error parsing XML file {filename}: {e}")
        return []
    except FileNotFoundError:
        logging.error(f"The file {filename} was not found.")
        return []
    except Exception as e:
        logging.error(f"An unexpected error occurred while processing {filename}: {e}")
        return []

    filename_element = root.find("filename")
    image_name = filename_element.text if filename_element is not None else "Unknown"

    size = root.find("size")
    if size is not None:
        width_element = size.find("width")
        height_element = size.find("height")
        try:
            width = int(width_element.text) if width_element is not None else 0
            height = int(height_element.text) if height_element is not None else 0
        except ValueError:
            logging.warning(f"Invalid width or height in {filename}. Setting width and height to 0.")
            width = height = 0
    else:
        logging.warning(f"Size information missing in {filename}. Setting width and height to 0.")
        width = height = 0

    objects = root.findall("object")
    parser = []

    for obj in objects:
        name_element = obj.find("name")
        name = name_element.text if name_element is not None else "Unknown"

        bndbox = obj.find("bndbox")
        if bndbox is not None:
            xmin_element = bndbox.find("xmin")
            xmax_element = bndbox.find("xmax")
            ymin_element = bndbox.find("ymin")
            ymax_element = bndbox.find("ymax")

            try:
                xmin = int(xmin_element.text) if xmin_element is not None else 0
                xmax = int(xmax_element.text) if xmax_element is not None else 0
                ymin = int(ymin_element.text) if ymin_element is not None else 0
                ymax = int(ymax_element.text) if ymax_element is not None else 0
            except ValueError:
                logging.warning(f"Invalid bounding box coordinates in {filename}. Setting coordinates to 0.")
                xmin = xmax = ymin = ymax = 0
        else:
            logging.warning(f"Bounding box information missing for an object in {filename}. Setting coordinates to 0.")
            xmin = xmax = ymin = ymax = 0

        parser.append([image_name, width, height, name, xmin, xmax, ymin, ymax])

    return parser


def label_encoding(label):
    """
    Converts object labels to numerical IDs for model training.

    This function maps each object label to a unique numerical identifier
    as defined in the labels' dictionary.

    Parameters
    ----------
    label : str
        The name of the object label to be encoded.

    Returns
    -------
    int
        The numerical ID corresponding to the object label.

        Returns -1 if the label is not found in the predefined dictionary.
    """
    labels = {
        'person': 0,'bicycle': 1, 'garbage container': 2, 'bus': 3, 'car': 4, 'train': 5, 'traffic_sign': 6,
        'traffic cone': 7, 'rail': 8, 'pothole': 9, 'pole': 10, 'downstair': 11,
        'upstair': 12, 'curb': 13, 'truck': 14, 'green-light': 15, 'red-light': 16,
        'crosswalk-sign': 17, 'crosswalk': 18, 'chair': 19, 'bottle': 20, 'pottedplant': 21,
        'bird': 22, 'dog': 23 ,'sofa': 24 ,'horse': 25, 'boat': 26, 'motorbike': 27,
        'cat': 28, 'tvmonitor': 29, 'cow': 30, 'sheep': 31, 'aeroplane': 32, 'diningtable': 33,
        'Tree': 34, 'Ponds': 35
    }
    try:
        return labels[label]
    except KeyError:
        logging.warning(f"Label '{label}' not found in label encoding dictionary. Assigning -1.")
        return -1


def save_data(filename, folder_path, groupby_obj):
    """
    Moves an image file and its corresponding label file to a specified directory.

    This function performs the following operations:
    1. Moves the image file from the "data_images" directory to the provided `folder_path`.
    2. Retrieves the associated label data from the `groupby_obj` (a pandas GroupBy object).
    3. Saves the label data as a `.txt` file in the `folder_path` with the same base name as the image.

    Parameters
    ----------
    filename : str
        The name of the image file to be moved (e.g., 'image1.jpg').
    folder_path : str
        The destination directory where the image and label files will be moved.
    groupby_obj : pandas.core.groupby.generic.DataFrameGroupBy
        A pandas GroupBy object that groups the data by 'filename'. It is used to retrieve
        the label data corresponding to the given `filename`.

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If the source image or label file does not exist.
    Exception
        If an error occurs during the file move or label file saving process.
    """
    try:
        src = os.path.join("data_images", filename)
        dst = os.path.join(folder_path, filename)
        move(src, dst)
        logging.info(f"Moved image: {src} to {dst}")
    except FileNotFoundError as e:
        logging.error(f"File not found while moving image: {e}")
        return
    except Exception as e:
        logging.error(f"Error moving image {filename}: {e}")
        return

    try:
        text_filename = os.path.join(
            folder_path,
            os.path.splitext(filename)[0] + ".txt"
        )
        group = groupby_obj.get_group(filename)
        if group.empty:
            logging.warning(f"No label data found for {filename}.")
            return
        group.set_index("filename").to_csv(
            text_filename, sep=" ", index=False, header=False
        )
        logging.info(f"Saved label file: {text_filename}")
    except KeyError:
        logging.error(f"Label data for {filename} not found in the groupby object.")
    except Exception as e:
        logging.error(f"Error saving label file for {filename}: {e}")


parser_all = list(map(extract_text, xml_list))
try:
    data = reduce(lambda x, y: x + y, parser_all, [])
except TypeError as e:
    logging.error(f"Error during list flattening with reduce: {e}")
    data = []

if data:
    try:
        df = pd.DataFrame(data, columns=["filename", "width", "height", "name", "xmin", "xmax", "ymin", "ymax"])
        logging.info("DataFrame created successfully.")
    except Exception as e:
        logging.error(f"Error creating DataFrame: {e}")
        df = pd.DataFrame(columns=["filename", "width", "height", "name", "xmin", "xmax", "ymin", "ymax"])
else:
    df = pd.DataFrame(columns=["filename", "width", "height", "name", "xmin", "xmax", "ymin", "ymax"])
    logging.warning("No data to create DataFrame.")

cols = ['width', 'height', 'xmin', 'xmax', 'ymin', 'ymax']
if not df.empty:
    try:
        df[cols] = df[cols].astype(int)
    except ValueError as ve:
        logging.error(f"Error converting columns to int: {ve}")

    try:
        df['center_x'] = ((df['xmin'] + df['xmax']) / 2) / df['width']
        df['center_y'] = ((df['ymin'] + df['ymax']) / 2) / df['height']
        df['w'] = (df['xmax'] - df['xmin']) / df['width']
        df['h'] = (df['ymax'] - df['ymin']) / df['height']
    except ZeroDivisionError as zde:
        logging.error(f"Division by zero error during coordinate calculations: {zde}")
        df['center_x'] = df['center_y'] = df['w'] = df['h'] = 0

else:
    logging.warning("DataFrame is empty. No calculations performed.")

if not df.empty:
    images = df['filename'].unique()
    img_df = pd.DataFrame(images, columns=['filename'])

    try:
        img_train = tuple(img_df.sample(frac=0.8, random_state=42)['filename'])
        img_test = tuple(img_df.query('filename not in @img_train')['filename'])
        logging.info(f"Number of training images: {len(img_train)}")
        logging.info(f"Number of testing images: {len(img_test)}")
    except Exception as e:
        logging.error(f"Error during train-test split: {e}")
        img_train = ()
        img_test = ()
else:
    img_train = ()
    img_test = ()
    logging.warning("No images available for training and testing splits.")

try:
    train_df = df[df['filename'].isin(img_train)].copy()
    test_df = df[df['filename'].isin(img_test)].copy()
except Exception as e:
    logging.error(f"Error filtering DataFrame for train and test sets: {e}")
    train_df = pd.DataFrame(columns=df.columns)
    test_df = pd.DataFrame(columns=df.columns)

try:
    train_df["id"] = train_df["name"].apply(label_encoding)
    test_df["id"] = test_df["name"].apply(label_encoding)
except Exception as e:
    logging.error(f"Error during label encoding: {e}")

train_folder = "data_images/train1"
test_folder = "data_images/test1"

for folder in [train_folder, test_folder]:
    try:
        os.makedirs(folder, exist_ok=True)
        logging.info(f"Directory '{folder}' created or already exists.")
    except Exception as e:
        logging.error(f"Error creating directory '{folder}': {e}")

cols = ['filename', 'id', 'center_x', 'center_y', 'w', 'h']

if not train_df.empty:
    groupby_obj_train = train_df[cols].groupby('filename')
else:
    groupby_obj_train = pd.DataFrame().groupby('filename')

if not test_df.empty:
    groupby_obj_test = test_df[cols].groupby('filename')
else:
    groupby_obj_test = pd.DataFrame().groupby('filename')

filename_series_train = pd.Series(groupby_obj_train.groups.keys())
filename_series_test = pd.Series(groupby_obj_test.groups.keys())
filename_series_train.apply(save_data, args=(train_folder, groupby_obj_train))
filename_series_test.apply(save_data, args=(test_folder, groupby_obj_test))

logging.info("Dataset split into train and test sets successfully.")