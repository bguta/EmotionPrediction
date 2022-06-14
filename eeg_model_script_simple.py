
# EEG Model Training Notebook
# 
# This notebook contains the model training pipeline used for EEG classification. An overview of this notebook is as follows
# 
# 1. Training/Testing dataset creation
# 2.Simple Classifier
# 3. Overall Results

# import some useful libraries

import os
import glob

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import pandas as pd

from scipy import signal

# model creation
import torch.nn as nn
import torch.nn.functional as F
import torch

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold

# loading bar
from tqdm.auto import tqdm

# timer
import time

# Training/Testing dataset creation
# functions for preprocessing dataset
# the data is stored in an Nx66 matrix. The first column is time in milliseconds, the second is the min/max normalized feel trace ([0,1])
# the other 64 entries are the eeg channels 

def load_and_split_dataset(eeg_ft_dir = 'ALIGNED_DATA', split_size=100, subject_num = 5, k=5, label_type='angle', num_classes=3):
    # choose the subject
    subject_data_files = glob.glob(os.path.join(eeg_ft_dir, '*.csv'))
    # sort the files by the index given to them
    file_name_2_index = lambda file : int(file.split('.csv')[0].split('_')[-1])
    subject_data_files.sort() # sort alphabetically
    subject_data_files.sort(key=file_name_2_index) # sort by index
    eeg_ft = subject_data_files[subject_num-1]

    print(f"Chosen subject: {eeg_ft}")
    
    input_label_pair = pd.read_csv(eeg_ft).values # read the Nx66 data for a single subject

    x = ( input_label_pair[:,2:] - input_label_pair[:,2:].min(axis=0, keepdims=True) )
    y = ( input_label_pair[:,2:].max(axis=0, keepdims=True) -  input_label_pair[:,2:].min(axis=0, keepdims=True) )
    input_label_pair[:,2:] = x/y # eeg-channel-wise min/max normalization

    dataset = [input_label_pair[x : x + split_size] for x in range(0, len(input_label_pair), split_size)] # split into windows here
    if len(dataset[-1]) < split_size:
        dataset.pop() # remove last window if it is smaller than the rest

    if label_type != 'both':
        labels = get_label(dataset, n_labels=num_classes, label_type=label_type).squeeze() # (N, 1)
    else:
        labels = get_combined_label(dataset, n_labels=int(np.sqrt(num_classes))).squeeze() # (N, 1)
    
    # generate features from eeg signal
    dataset = generate_eeg_features(dataset, split_size=split_size)
    dataset = np.vstack([np.expand_dims(x,0) for x in dataset]) # (N, eeg_feature_size, 64, 64)
    print(f"EEG features (N, eeg_feature_size, 64, 64):  {dataset.shape}")
    print(f"label set shape (N,):  {labels.shape}")

    indices = split_dataset(labels, k=k) # split data into train/test indices using kFold validation
    return dataset, labels, indices


def get_label(data, n_labels=3, label_type='angle'):
    if label_type == 'angle':
        labels = stress_2_angle(np.vstack([x[:,1].T for x in data])) # angle/slope mapped to [0,1] in a time window
    elif label_type == 'pos':
        labels = np.vstack([x[:,1].mean() for x in data]) # mean value within the time window
    else:
        labels = stress_2_accumulator(np.vstack([x[:,1].T for x in data])) # accumulator mapped to [0,1] in a time window
        
    label_dist = stress_2_label(labels, n_labels=n_labels)
    return label_dist

def get_combined_label(data, n_labels=3):
    angle_labels = get_label(data, n_labels=n_labels, label_type='angle').squeeze() # (N, 1)
    pos_labels = get_label(data, n_labels=n_labels, label_type='pos').squeeze() # (N, 1)

    labels = [x for x in range(n_labels)]
    labels_dict =  {(a, b) : n_labels*a+b for a in labels for b in labels} # cartesian product
    combined_labels = [labels_dict[(pos, angle)] for (pos, angle) in zip(pos_labels, angle_labels)]
    return np.array(combined_labels)


def stress_2_label(mean_stress, n_labels=5):
    # value is in [0,1] so map to [0,labels-1] and discretize
    return np.digitize(mean_stress * n_labels, np.arange(n_labels)) - 1

def stress_2_angle(stress_windows):
    '''
    do a linear least squares fit in the time window
    stress_window: (N_samples, time_window)
    '''
    xvals = np.arange(stress_windows.shape[-1])/1e3/60 # time in (minutes)
    slope = np.polyfit(xvals, stress_windows.T, 1)[0] # take slope linear term # 1/s
    angle = np.arctan(slope)/ (np.pi/2) * 0.5 + 0.5 # map to [0,1]
    return angle

def stress_2_accumulator(stress_windows):
    '''
    apply an integral to the time window
    stress_window: (N_samples, time_window)
    '''
    max_area = stress_windows.shape[-1]
    xvals = np.arange(stress_windows.shape[-1]) # time in (ms)
    integral = np.trapz(stress_windows, x=xvals)
    return integral/max_area # map to [0,1]

def split_dataset(labels, k=5):
    '''
    split the features and labels into k groups for k fold validation
    we use StratifiedKFold to ensure that the class distrubutions within each sample is the same as the global distrubution
    '''
    kf = StratifiedKFold(n_splits=k, shuffle=True)

    # only labels are required for generating the split indices so we ignore it
    temp_features = np.zeros_like(labels)
    indices = [(train_index, test_index) for train_index, test_index in kf.split(temp_features, labels)]
    return indices

def generate_eeg_features(dataset, split_size=200):
    sample_freq = 1000
    # get FFT
    psd_windows = [signal.periodogram(x[:,2:], sample_freq, axis=0) for x in dataset ] # get the power spectral density for each window

    # frequency bands
    bands={'alpha': (8, 12), 'beta': (12, 30), 'delta': (1, 4), 'theta': (4, 7), 'gamma': (30, 50)}
    band_freqs = [bands[x] for x in ['alpha', 'beta', 'delta', 'theta', 'gamma']]

    features = []
    for window in psd_windows: # calculate the power in each band for channel for each window
        freqs, psd = window
        idx_bands = [np.logical_and(freqs >= low, freqs <= high) for low,high in band_freqs]

        freq_res = freqs[1] - freqs[0]
        band_powers = np.array([sp.integrate.simpson(psd[idx,:], dx=freq_res, axis=0) for idx in idx_bands]) # (5,64)
        total_powers = np.array([sp.integrate.simpson(psd, dx=freq_res, axis=0) for idx in idx_bands]) # (5,64)
        diff_entropy = -0.5 * np.log(band_powers/total_powers)
        # (5, 1, 64)
        # (5, 64, 1)
        diff_de = np.expand_dims(diff_entropy, axis=2) - np.expand_dims(diff_entropy, axis=1) # (5,64,64)
        diff_de = (diff_de  - diff_de.min(axis=(1,2), keepdims=True))/(diff_de.max(axis=(1,2), keepdims=True) - diff_de.min(axis=(1,2), keepdims=True))
        
        features.append(diff_de)
    return features

# helper function
class classifier(nn.Module):
    def __init__(self, num_features=12, num_hidden=32, dropout=0.2, n_labels=5):
        super(classifier, self).__init__()
        
        self.hidden_size = num_hidden
        self.input_size = num_features
        self.n_classes = n_labels


        self.cnn = nn.Sequential(
            nn.Conv2d(5, 8, 3, padding='same'),
            nn.ReLU()
        )
        
        self.classify = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(64 * 64 * 8, self.n_classes))
    
    def forward(self,x):
        x = self.cnn(x)
        x = self.classify(x) 
        return x

class classifier_dataset(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        'Initialization'
        self.x = features # (N, eeg_feature_size, 64, 64)
        self.labels = labels # (N, 1)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.x)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        x = torch.from_numpy(self.x[index]).float() # (eeg_feature_size, 64, 64)
        y = torch.from_numpy(np.array(self.labels[index])).long() # feel trace labels int value [0,n_labels]
        return x, y


def train_classifier(model, num_epochs=5, batch_size=1, learning_rate=1e-3, features=None, labels=None, num_classes=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    criterion = nn.CrossEntropyLoss()
    train_dataset = classifier_dataset(features, labels)
    
    # figure out class distribution to over sample less represented classes

    train_labels = labels
    
    # get the weights of each class as 1/occurrence
    train_class_weight = np.bincount(train_labels, minlength=num_classes)
    print(f"Train label distribution: {train_class_weight}")
    train_class_weight = 1/train_class_weight
    
    # get the per sample weight, which is the likelihood os sampling
    train_sample_weights = [train_class_weight[x] for x in train_labels]
    
    # sampler, weighted by the inverse of the occurrence
    train_sampler = torch.utils.data.WeightedRandomSampler(train_sample_weights, len(train_sample_weights), replacement=True)
    
        
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size=batch_size,
                                               num_workers=8,
                                               sampler=train_sampler)
    
    train_metrics = []
    for epoch in range(num_epochs):
        
        # reset metrics
        cur_train_acc = 0 # accuracy
        cur_train_pc = 0 # precision
        cur_train_rc = 0 # recall
        cur_train_f1 = 0 # f1
        cur_train_loss = 0 # loss
        
        # set to train mode
        model.train()
        
        # loop over dataset
        for data in train_loader:
            x, y = data
            x = x.to(device)
            y = y.to(device)
            
            y_hat = model(x)
            loss = criterion(y_hat, y)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            y_hat_np = F.softmax(y_hat.detach(), dim=1).argmax(axis=1).cpu().numpy().squeeze().reshape(-1,) # predictions
            y_np = y.detach().cpu().numpy().squeeze().reshape(-1,) # labels
            
            # metrics
            prf = precision_recall_fscore_support(y_np, y_hat_np, average='macro', zero_division=0)
            
            cur_train_acc += np.mean(y_hat_np == y_np)
            cur_train_pc += prf[0]
            cur_train_rc += prf[1]
            cur_train_f1 += prf[2]
            cur_train_loss += loss.detach().cpu()
        
        # average metrics over loop
        train_loop_size = len(train_loader)
        cur_train_acc  = cur_train_acc/train_loop_size
        cur_train_pc   = cur_train_pc/train_loop_size
        cur_train_rc   = cur_train_rc/train_loop_size
        cur_train_f1   = cur_train_f1/train_loop_size
        cur_train_loss = cur_train_loss/train_loop_size
        
        
        train_metrics.append([cur_train_acc, cur_train_pc, cur_train_rc, cur_train_f1, cur_train_loss])
            
        # print(f'Epoch:{epoch+1},'\
        #       f'\nTrain Loss:{cur_train_loss},'\
        #       f'\nTrain Accuracy:{cur_train_acc},'\
        #       f'\nTrain Recall: {cur_train_rc},'\
        #       f'\nTrain precision: {cur_train_pc},' \
        #       f'\nTrain F1-Score:{cur_train_f1},')
        
    return train_metrics
        

def main_runner(subject_choice=1, label_type='angle'):

    # Before we create the models, we will first prepare the data by splitting it and preprocessing it.
    eeg_feeltrace_dir = 'ALIGNED_DATA' # directory containing *.csv files
    # hyper parameters
    window_size = 500 # must be an int in milliseconds
    subject_num = subject_choice # which subject to choose [1-16]
    k_fold = 5 # k for k fold validation
    #label_type = 'angle' # 'angle' or 'pos' or 'both'
    num_classes = 3 if label_type != 'both' else 9 # number of classes to discretize the labels into
    num_features = 64 # features for encoder
    classifier_learning_rate = 1e-3 # adam learning rate
    classifier_train_epochs = 30 # train classifier duration
    classifier_hidden = 8 # the larger the more complicated the model

    dataset, labels, indices = load_and_split_dataset(eeg_ft_dir = 'ALIGNED_DATA', split_size=window_size, subject_num = subject_num, k=k_fold, label_type=label_type, num_classes=num_classes)
    print(f"Label class bincount: {np.bincount(labels, minlength=num_classes)}")
    
    k_acc = [] # accuracies for each fold
    k_f1 = [] # f1 score for each fold
    k_cm = [] # confusion matrix for each fold

    for cur_k in range(len(indices)):
        print(f"Training k={cur_k}")
        train_index, test_index = indices[cur_k]
        classifier_model = classifier(num_features=num_features, num_hidden=classifier_hidden, dropout=0.8, n_labels=num_classes)

        encoded_train_features = dataset[train_index]

        print('Training Classifier!')
        classifier_train_metrics = train_classifier(classifier_model, classifier_train_epochs, batch_size=128, learning_rate=classifier_learning_rate, features=encoded_train_features, labels=labels[train_index], num_classes=num_classes)


        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        test_features, test_labels =  dataset[test_index], labels[test_index]
        print(f"Test label distribution: {np.bincount(test_labels, minlength=num_classes)}")
        encoded_test_features = test_features

        with torch.no_grad():
            classifier_model.eval()
            x_encoded  = torch.from_numpy(encoded_test_features).float().to(device)
            y = test_labels
            y_hat = classifier_model(x_encoded)
            y_hat = F.softmax(y_hat.detach(), dim=-1).cpu().numpy()
            preds = y_hat

        # fig, axs = plt.subplots(figsize=(20,20), dpi=120)
        # axs.set_title(f"LSTM Feel Trace Model Confusion Matrix - Emotion-as-{label_type}", fontsize=20)
        # axs.set_xlabel("Predicted Label", fontsize=15)
        # axs.set_ylabel("True Label", fontsize=15)


        prf = precision_recall_fscore_support(test_labels, np.array([x.argmax() for x in preds]), average='macro', zero_division=0)
        acc = np.mean(test_labels == np.array([x.argmax() for x in preds]))
        print(f"Precision: {prf[0]}")
        print(f"Recall: {prf[1]}")
        print(f"F1-Score: {prf[2]}")
        print(f"Accuracy: {acc}")

        k_acc.append(acc)
        k_f1.append(prf[2])

        cm = confusion_matrix(test_labels, [x.argmax() for x in preds], labels=np.arange(num_classes), normalize=None)
        #k_cm.append(cm)
        #disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(num_classes))

        #disp.plot(ax=axs)
        #plt.show()

    print(f"Accuracy, Average accuracy: {k_acc}, {np.mean(k_acc)}")
    print(f"F1-Score, Average F1-Score: {k_f1}, {np.mean(k_f1)}")
    p_label = f'P0{subject_choice}' if subject_choice < 10 else f'P{subject_choice}'
    ret = [p_label] + [x for x in np.bincount(labels, minlength=num_classes)] + [label_type, window_size, 'EEG', np.mean(k_acc), np.mean(k_f1), np.std(k_acc), np.std(k_f1)]
    return ret

def run():
    subjects = [(x+1) for x in range(16)]
    t0 = time.time()
    
    label_type = 'pos'
    result_list_1 = np.vstack([main_runner(subject, label_type) for subject in tqdm(subjects)])
    label_type = 'angle'
    result_list_2 = np.vstack([main_runner(subject, label_type) for subject in tqdm(subjects)])
    label_type = 'accumulator'
    result_list_3 = np.vstack([main_runner(subject, label_type) for subject in tqdm(subjects)])

    result_list = np.vstack([result_list_1, result_list_2, result_list_3])
    t1 = time.time()
    print(f"Total time (s): {t1-t0}")
    if label_type != 'both':
        result_df = pd.DataFrame(result_list, columns=['Participant', 'Class 1', 'Class 2', 'Class 3', 'Label Type', 'Window [ms]', 'Modality', 'Accuracy', 'F1-Score', 'STDEV Accuracy', 'STDEV F1-Score'])
    else:
        result_df = pd.DataFrame(result_list, columns=['Participant', 'Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 6', 'Class 7', 'Class 8', 'Class 9', 'Label Type', 'Window [ms]', 'Modality', 'Accuracy', 'F1-Score', 'STDEV Accuracy', 'STDEV F1-Score'])

    result_df.to_csv('eeg_classification_result_simple.csv', index=False)


if __name__ == '__main__':
    run()
