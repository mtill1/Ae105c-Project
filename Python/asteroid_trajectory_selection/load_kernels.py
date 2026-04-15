"""Load SPICE kernels and extract asteroid IDs from BSP files."""

import os
import glob
import spiceypy


def load_kernels(bsp_folder_name, generic_kernels_path):
    """Load generic + project-specific SPICE kernels, return asteroid list.

    Parameters
    ----------
    bsp_folder_name : str
        Path to folder containing asteroid .bsp files.
    generic_kernels_path : str
        Path to directory containing generic SPICE kernels
        (with subdirectories lsk/, spk/satellites/, spk/planets/, pck/).

    Returns
    -------
    asteroid_list : list of dict
        Each dict has 'ID' (int or list of int) and 'NAME' (str) keys.
    """
    # Load universal kernels
    spiceypy.furnsh(os.path.join(generic_kernels_path, 'lsk', 'naif0012.tls'))
    spiceypy.furnsh(os.path.join(generic_kernels_path, 'spk', 'satellites', 'jup310.bsp'))
    spiceypy.furnsh(os.path.join(generic_kernels_path, 'spk', 'planets', 'de430.bsp'))
    spiceypy.furnsh(os.path.join(generic_kernels_path, 'pck', 'gm_de431.tpc'))
    spiceypy.furnsh(os.path.join(generic_kernels_path, 'pck', 'pck00010.tpc'))

    # Load project-specific kernels (asteroids)
    bsp_files = sorted(glob.glob(os.path.join(bsp_folder_name, '*.bsp')))

    asteroid_list = []
    for bsp_path in bsp_files:
        spiceypy.furnsh(bsp_path)

        # Extract NAIF IDs from the SPK file
        ids = spiceypy.spkobj(bsp_path)

        # File name without extension
        name = os.path.splitext(os.path.basename(bsp_path))[0]

        asteroid_list.append({
            'ID': ids,
            'NAME': name,
        })

    return asteroid_list
