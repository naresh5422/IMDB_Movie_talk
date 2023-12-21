import os
from box.exceptions import BoxValueError
import yaml
from IMDBClassifier import logger
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import base64

@ensure_annotations
def read_yaml(path_to_yaml:Path) -> ConfigBox:
    """
    Read YAML file and return
    Args:
        path_to-yaml: (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returs:
        ConfigBox: ConfigBox type
    """

    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("YAML file is empty")
    except Exception as e:
        raise e


@ensure_annotations
def create_directories(path_to_directories:list, verbose = True):
    """
        Create List of directories

        Args:
        path_to_directories (list): path of directories list
        ignore_log(bool, optional):
        ignore if multiple directories are to be created defaults to False
    """

    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Created directory at: {path}")


@ ensure_annotations
def save_json(path: Path, data: dict):
    """ Save the JSON data

    Args:
        path (Path): path to JSON file
        data (dict); data to be saved into JSON file format
    """

    with open(path, "w") as f:
        json.dump(data, f, indent = 4)

    logger.info(f"Now JSON file format saved at: {path}")



@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """ Load json file data
    Args:
        path(Path): json file path

    Returns:
        ConfigBoc: data as class attributes instead of dict

        """

    with open(path) as f:
        content = json.load(f)
    logger.info(f"JSON file loaded successfully from: {path}")
    return ConfigBox(content)

@ensure_annotations
def save_bin(data: Any, path: Path):
    """
    save binary file

    Args:
        data (Any): data to be saved as binary
        path (Path): path to binary file

    """

    joblib.dump(value = data, filename=path)
    logger.info(f"Binary file saved at: {path}")


@ensure_annotations
def load_bin(path: Path) -> Any:
    """
    Load binary data

    Args:
        path (Path): path to binary file

    Return:
        Any: Object stored in the file
    """

    data = joblib.load(path)
    logger.info(f"Binary file loaded from: {path}")
    return data


@ensure_annotations
def get_size(path: Path) -> str:
    """
    Get size in KB

    Args:
        path (Path): file path

    Return:
        str: file size in KB
    """

    size_in_KB = round(os.path.getsize(path)/1024)
    return f"~ {size_in_KB} KB"

def decodeImage(imgstring, fileName):
    imgdata = base64.b64decode(imgstring)
    with open(fileName, "wb") as f:
        f.write(imgdata)
        f.close()

def encodeImageIntoBase64(croppedImagePath):
    with open(croppedImagePath, "rb") as f:
        return base64.b64decode(f.read())
