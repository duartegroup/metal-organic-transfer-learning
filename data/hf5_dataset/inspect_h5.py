import h5py
import sys
import numpy as np

file_path = sys.argv[1]


def print_hdf5_structure_recursive(obj, indent_level=0):
    """
    A recursive function to print the structure of an HDF5 file.
    """
    indent = "  " * indent_level

    if isinstance(obj, h5py.Group):
        print(f"{indent}Group: {obj.name.split('/')[-1] if obj.name != '/' else '/'}")
        # Print attributes of the group
        if obj.attrs:
            print(f"{indent}  Attributes:")
            for attr_name, attr_value in obj.attrs.items():
                print(f"{indent}    - {attr_name}: {attr_value}")

        # Recursively visit members of the group
        for key in obj.keys():
            print_hdf5_structure_recursive(obj[key], indent_level + 1)

    elif isinstance(obj, h5py.Dataset):
        print(f"{indent}Dataset: {obj.name.split('/')[-1]}")
        print(f"{indent}  Shape: {obj.shape}")
        print(f"{indent}  Dtype: {obj.dtype}")
        # Print attributes of the dataset
        if obj.attrs:
            print(f"{indent}  Attributes:")
            for attr_name, attr_value in obj.attrs.items():
                print(f"{indent}    - {attr_name}: {attr_value}")
    else:
        print(f"{indent}Unknown HDF5 object type: {type(obj)}")


try:
    with h5py.File(file_path, "r") as f:
        print(f"Inspecting HDF5 file: {file_path}")
        print_hdf5_structure_recursive(f)  # Start recursion from the root file object

except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
except Exception as e:
    print(f"An error occurred: {e}")
