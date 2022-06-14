import pandas as pd
import numpy as np
import scipy.io as sp_io
import os
import glob
import contextlib
import joblib
from tqdm import tqdm
from joblib import Parallel, delayed



@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def normalize_and_crop(eeg: np.array, ft: np.array) -> tuple:
    """
    :param eeg:
    :param ft:
    :return:
    """
    min_index = np.where(ft[:, 0] >= 0)[0][0]  # first positive time stamp
    max_index = int(min(eeg[:, 0][-2], ft[:, 0][-1]))
    # find the index on ft that is greater than or equal to the max index timestamp
    ft_end_index = np.where(ft[:, 0] > max_index)[0][0] - 1 if ft[:, 0][-1] > eeg[:, 0][-2] else \
        np.where(ft[:, 0] == max_index)[0][0]

    new_ft = ft[min_index:ft_end_index + 1, :].copy()
    new_eeg = eeg[int(ft[:, 0][min_index]):max_index + 1, :-1].copy()

    # normalize to be between [0,1]
    min_eeg = np.min(new_eeg[:, 1:], axis=0)
    max_eeg = np.max(new_eeg[:, 1:], axis=0)

    min_ft = 0
    max_ft = 225

    new_eeg[:, 1:] = (new_eeg[:, 1:] - min_eeg) / (max_eeg - min_eeg)
    new_ft[:, 1] = (new_ft[:, 1] - min_ft) / (max_ft - min_ft)

    new_ft[:, 0] = new_ft[:, 0] / 1000  # ms to seconds
    new_eeg[:, 0] = new_eeg[:, 0] / 1000  # ms to seconds

    return new_eeg, new_ft


def create_dataset(src_dir: str) -> None:
    """
    :param src_dir: the directory containing all the EEG and FeelTrace data in folders for each subject

    Creates the normalized and cropped dataset in the EEG_FT_DATA directory, throws an error if
    EEG_FT_DATA does not exist
    """
    subject_data_dir = glob.glob(os.path.join(src_dir, 'p*'))

    subject_data = [glob.glob(os.path.join(x, '*')) for x in subject_data_dir]
    all_eeg_data = [x[21] for x in subject_data]
    all_joystick_data = [x[15] for x in subject_data]

    # this next step takes a bit of time!!
    eeg_ft_pairs = [(eeg, ft) for eeg, ft in zip(all_eeg_data, all_joystick_data)]

    with tqdm_joblib(tqdm(desc="Dataset Creation", total=len(eeg_ft_pairs))) as progress_bar:
        Parallel(n_jobs=8)(delayed(write_to_csv_dataset_loop)(i, x, y) for i, (x, y) in enumerate(eeg_ft_pairs))
    print('Done! Created dataset in EEG_FT_DATA')


def write_to_csv_dataset_loop(index: int, x: str, y: str) -> None:
    """
    Should not be called by the user, for pair at index, create the pandas dataframe and write to a csv file
    :param y: FeelTrace filename
    :param x: EEG filename
    :param index: index of the pair to write
    """

    eeg_column_headers = ['t'] + [f'channel_{i}' for i in range(64)]
    ft_column_headers = ['t', 'stress']

    eeg, ft = sp_io.loadmat(x)['var'], sp_io.loadmat(y)['var']
    normalized_eeg, normalized_ft = normalize_and_crop(eeg, ft)

    eeg_df = pd.DataFrame(data=normalized_eeg, columns=eeg_column_headers)
    ft_df = pd.DataFrame(data=normalized_ft, columns=ft_column_headers)

    eeg_df.to_csv(os.path.join('../EEG_FT_DATA', f'normalized_eeg_{index}.csv'), index=False)
    ft_df.to_csv(os.path.join('../EEG_FT_DATA', f'normalized_ft_{index}.csv'), index=False)
