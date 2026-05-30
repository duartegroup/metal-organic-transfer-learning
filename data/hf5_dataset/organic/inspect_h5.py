import h5py
import sys

file_path = sys.argv[1]

def get_group_storage_size(group):
    """Recursively compute total storage size (in bytes) of all datasets in this group."""
    total_size = 0
    for key in group.keys():
        item = group[key]
        if isinstance(item, h5py.Dataset):
            total_size += item.id.get_storage_size()
        elif isinstance(item, h5py.Group):
            total_size += get_group_storage_size(item)
    return total_size

def print_hdf5_structure_recursive(obj, indent_level=0):
    """
    Print HDF5 file structure with dataset shapes, types, and storage sizes in MB.
    Also print cumulative group size in MB.
    """
    indent = "  " * indent_level

    if isinstance(obj, h5py.Group):
        group_size_bytes = get_group_storage_size(obj)
        group_size_mb = group_size_bytes / (1024 * 1024)
        group_name = obj.name.split('/')[-1] if obj.name != '/' else '/'
        print(f"{indent}Group: {group_name}  [Total size: {group_size_mb:.3f} MB]")

        if obj.attrs:
            print(f"{indent}  Attributes:")
            for attr_name, attr_value in obj.attrs.items():
                print(f"{indent}    - {attr_name}: {attr_value}")

        for key in obj.keys():
            print_hdf5_structure_recursive(obj[key], indent_level + 1)

    elif isinstance(obj, h5py.Dataset):
        storage_bytes = obj.id.get_storage_size()
        storage_mb = storage_bytes / (1024 * 1024)
        print(f"{indent}Dataset: {obj.name.split('/')[-1]}")
        print(f"{indent}  Shape: {obj.shape}")
        print(f"{indent}  Dtype: {obj.dtype}")
        print(f"{indent}  Storage size: {storage_mb:.3f} MB")
        if obj.attrs:
            print(f"{indent}  Attributes:")
            for attr_name, attr_value in obj.attrs.items():
                print(f"{indent}    - {attr_name}: {attr_value}")

try:
    with h5py.File(file_path, "r") as f:
        print(f"Inspecting HDF5 file: {file_path}")
        print_hdf5_structure_recursive(f)
except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
except Exception as e:
    print(f"An error occurred: {e}")

