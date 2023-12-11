import os
import socket
import nbformat
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

def export_marked_cells(notebook_path, output_path):
    '''
    Exports code cells marked with #export in notebook_path to output_path
    Args:
        notebook_path:
        output_path:

    Returns:

    '''
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    export_cells = []
    for cell in nb.cells:
        if cell.cell_type == 'code' and cell.source.startswith('#export'):
            export_cells.append(cell.source)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(export_cells))

if __name__ == '__main__':
    base_dir = get_base_dir()
    nb_path = Path(base_dir / 'scgpt/j_scgpt_utils/cell_emb_plot.ipynb')
    output_path = Path(Path(base_dir / 'scgpt/j_scgpt_utils/cell_emb_plot.txt'))
    export_marked_cells(nb_path, output_path)