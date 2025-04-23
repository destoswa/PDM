import pdal
import os
import numpy as np
import json


def convert_las_to_laz(in_las, out_laz, verbose=True):
    """
    Convert a LAS file to a LAZ file, stripping all extra dimensions.

    Parameters:
    - in_las: str, path to the input .las file
    - out_laz: str, path to the output .laz file
    - verbose: bool, whether to print a success message

    Returns:
    - None
    """
    pipeline_json = {
        "pipeline": [
            {
                "type": "readers.las",
                "filename": in_las,
            },
            {
                "type": "writers.las",
                "filename": out_laz,
                "compression": "laszip",  # Ensure compression to LAZ
                "minor_version": 2,  # LAS 1.2
                "dataformat_id": 3   # Point format 3 (has RGB + GPS time)
                # "extra_dims": "none"
            }
        ]
    }

    # Create and execute the pipeline
    pipeline = pdal.Pipeline(json.dumps(pipeline_json))
    pipeline.execute()

    if verbose:
        print(f"LAZ file saved at {out_laz}")


def main():
    print(os.getcwd())
    src_file = r"./data/temp/group_3008954.las"
    out_file = r"./data/temp/group_3008954.laz"
    convert_las_to_laz(src_file, out_file)


if __name__ == '__main__':
    main()
