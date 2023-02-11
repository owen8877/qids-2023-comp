from pathlib import Path


def ensure_dir(path: str):
    """
    Make sure that the given path is a valid directory by creating one if missing.

    :param path: the absolute/relative path to the desired directory
    :return:
    """
    Path(path).mkdir(parents=True, exist_ok=True)
