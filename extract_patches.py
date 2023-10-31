#!/usr/bin/env python3.9

"""extract_patches.py

Patch extraction script.
"""

import re
import glob
import os
from tqdm import tqdm
import pathlib

import numpy as np

from misc.patch_extractor import PatchExtractor
from misc.utils import rm_n_mkdir, parse_json_file

from dataset import get_dataset

import logging
import argparse
import sys
sys.path.append('/storage/homefs/jg23p152/project/hover_net')

def create_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    # Write logs to the SLURM output file
    file_handler = logging.StreamHandler(sys.stdout)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

def divide_list(lst, n):
    division_size = len(lst) // n
    divisions = [lst[i * division_size:(i + 1) * division_size] for i in range(n - 1)]
    divisions.append(lst[(n - 1) * division_size:])
    return divisions


def get_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument('--path_to_files', type=str) # path to json with list of dict with mat and png files
    parser.add_argument('--dataset_name', type=str, default='lizard') # lizard, pannuke, or tcga
    parser.add_argument('--root_dir', type=str, default='/storage/homefs/jg23p152/project') # root directory for saving patches
    parser.add_argument('--types_to_keep', type=str, default='2') # cell types to keep separated by commas


    args = parser.parse_args()
    args.types_to_keep = [int(i) for i in args.types_to_keep.split(",")]

    return args

# -------------------------------------------------------------------------------------
if __name__ == "__main__":

    logger = create_logger()
    args = get_parser()

    # SLURM job array variables
    slurm_job = 'slurm job array' if os.environ.get('SLURM_JOB_ID') else 'local machine'
    slurm_array_task_count = int(os.environ.get('SLURM_ARRAY_TASK_COUNT', 1))
    slurm_array_job_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
    logger.info(f'Script started. Running in {slurm_job}. Task ID: {slurm_array_job_id+1} out of {slurm_array_task_count}')


    # Determines whether to extract type map (only applicable to datasets with class labels).
    type_classification = True

    win_size = [540, 540] # patch size
    step_size = [164, 164] # There is overlap between patches. This chooses the sliding window size.
    extract_type = "mirror"  # Choose 'mirror' or 'valid'. 'mirror'- use padding at borders. 'valid'- only extract from valid regions.
    

    # Name of dataset - use lizard, pannuke, or tcga.
    # This used to get the specific dataset img and ann loading scheme from dataset.py
    dataset_name = args.dataset_name
    save_root = args.root_dir
    
    # a dictionary to specify where the dataset path should be
    dataset_info = parse_json_file(args.path_to_files)
    dataset_info = divide_list(dataset_info, slurm_array_task_count)[slurm_array_job_id]

    patterning = lambda x: re.sub("([\[\]])", "[\\1]", x)
    parser = get_dataset(dataset_name)
    xtractor = PatchExtractor(win_size, step_size)

    logger.info(f'Patch Extraction will commence for {len(dataset_info)} samples with patch dimensions: {win_size[0]}x{win_size[1]} and step size: {step_size[0]}x{step_size[1]}')
    pbar_format = "Process File: |{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}]"
    pbarx = tqdm(total=len(dataset_info), bar_format=pbar_format, ascii=True, position=0)

    for i in range(len(dataset_info)):
        img_file = os.path.join(save_root, dataset_info[i]['img_file'])
        ann_file = os.path.join(save_root, dataset_info[i]['mat_file'])

        sample_name = os.path.splitext(os.path.basename(img_file))[0]

        out_dir = os.path.join(os.path.dirname(img_file), 'patches', sample_name)

        img = parser.load_img(img_file)
        ann = parser.load_ann(ann_file, 
                              type_classification, 
                              args.types_to_keep, 
                              is_healthy=True if 'tiles_healthy' in img_file else False)

        img = np.concatenate([img, ann], axis=-1)

        rm_n_mkdir(out_dir)
        sub_patches, row0, row1, col0, col1 = xtractor.extract(img, extract_type)

        logger.info(f'Sample name: {sample_name}, with {len(sub_patches)} patches.')

        pbar_format = "Extracting  : |{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}]"
        pbar = tqdm(total=len(sub_patches),
                    leave=False,
                    bar_format=pbar_format,
                    ascii=True,
                    position=1)

        for idx, (patch, r0, r1, c0, c1) in enumerate(zip(sub_patches, row0, row1, col0, col1)):
            pbar.update()
            file_path = os.path.join(out_dir, f'id_{idx}_x_{r0}_{r1}_y_{c0}_{c1}.npy')
            # if os.path.exists(file_path):
            #     continue
            np.save(file_path, patch)
            
        pbar.close()

        pbarx.update()
    pbarx.close()

    logger.info('Script finished. All patches have been extracted!!')