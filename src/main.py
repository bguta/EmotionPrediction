from utils import create_dataset, save_aligned_dataset
import os

if __name__ == '__main__':
    create_dataset(os.path.join('..', 'RAW_EEG'), os.path.join('..', 'EEG_FT_DATA'), num_workers=2)
    save_aligned_dataset(os.path.join('..', 'EEG_FT_DATA'), os.path.join('..', 'ALIGNED_DATA'))
