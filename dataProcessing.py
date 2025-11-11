import pandas as pd
import numpy as np
import os
from pathlib import Path

#it goes [age]_[gender 0 = male 1 = female]

def _parse_filename(filename):
    parts = filename.split('_')
    age = int(parts[0])
    gender = int(parts[1])
    return age, gender


def _get_all_files_recursive(directory_path):
    all_files = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            all_files.append(os.path.join(root, file))
    return all_files



def get_data_df():
    files = _get_all_files_recursive('./UTKFace/')
    data = []

    for file in files:
        p = Path(file)
        filename = p.name
        age, gender = _parse_filename(filename)
        data.append({
            'file_path': str(file),
            'filename': filename,
            'age': age,
            'gender': gender
        })

    df = pd.DataFrame(data)
    return df
    
