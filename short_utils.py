import os
import socket
from pathlib import Path



def get_base_dir():
    # Define your mappings here
    dir_mapping = {
        'shairlab.upper.2080': '~/lab/trans_stamp/',
        'shair-DGX-Station': '/home/shair/Desktop/STAMP_2023/jesse/trans_stamp_curr/'
        # Add more mappings as needed
    }

    # Get the name of the computer
    hostname = socket.gethostname()

    # Check if the hostname is in your mapping
    if hostname in dir_mapping:
        # Return a Path object for the corresponding directory
        return Path(dir_mapping[hostname])
    else:
        #raise error
        raise ValueError(f"Hostname '{hostname}' not found in directory mapping")

