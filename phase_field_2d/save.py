# %%
import json
import yaml
import numpy as np
import datetime
import os
import shutil
from typing import Any

def create_directory(directory_name: str = "tmp", force: bool = False) -> None:
    """making new directory (name: default to "tmp")

    Args:
        directory_name (str, optional): directory name. Defaults to "tmp".
        force (bool, optional): make new directory and overwrite the existing directory even if the "directory_name" directory is exist. Defaults to False.
    """
    print("create_directory----------------------------------------")

    if os.path.exists(directory_name):
        if force:
            shutil.rmtree(directory_name)
            print(f"Removed existing directory: {directory_name}")
        else:
            print(f"The directory '{directory_name}' already exists.")
            print("skipping to create directory.")
            print("--------------------------------------------------------")
            return
            # raise Warning(f"The directory '{directory_name}' already exists.")

    os.makedirs(directory_name)
    print(f"Created a new directory: {directory_name}.")
    print("--------------------------------------------------------")


def save_str(path: str, string: str) -> None:
    """save the strings as "path" file.

    Args:
        path (str): _description_
        string (str): _description_
    """
    with open(path, "w") as file:
        file.write(string)


def make_dir_name(file_name: str = "output") -> str:
    """Automatically make directory name as string. The format is `{file_name}_ %Y-%m-%d-%H-%M-%S'

    Args:
        file_name (str, optional): The name of head. Defaults to "output".

    Returns:
        str: name
    """
    current_time = datetime.datetime.now()
    timestamp = current_time.strftime("%Y-%m-%d-%H-%M-%S")
    filename = f"{file_name}_{timestamp}"
    return filename


def get_base_name(str: str) -> str:
    """get base name of file. e.g., xxxx.png -> xxxx

    Args:
        str (str): input

    Returns:
        str: base name
    """
    basename = os.path.basename(str)
    return basename.split(".")[0]


def instance_to_dict(instance: Any, properties_list: list[str]) -> dict:
    """make dictionary of instance properties. Only properties in `properties_list' is used to make dictionary.
    j
        Args:
            instance (Any): instance of class
            properties_list (list[str]): _description_

        Returns:
            dict: _description_
    """
    return {property: instance.__dict__[property] for property in properties_list}


def dump(instance_dict: dict) -> str:
    """same as yaml.dump

    Args:
        instance_dict (dict): _description_

    Returns:
        str: _description_
    """
    return yaml.dump(instance_dict)


if __name__ == "__main__":
    create_directory("tmp")
    dirname = make_dir_name()
    create_directory(f"tmp/{dirname}", force=True)
    print(get_base_name("result/output_2023-12-06-19-43-56/con_3330.npy"))

    class MyClass:
        def __init__(self, name: str, age: int, location: str):
            self.name = name
            self.age = age
            self.location = location

    my_instance = MyClass("John Doe", 30, "New York")
    instance_dict = instance_to_dict(my_instance, ["name", "age"])
    yaml_str = yaml.dump(instance_dict)
    save_str("tmp/test.yaml", yaml_str)


# %%
