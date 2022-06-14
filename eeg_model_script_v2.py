
# EEG Model Training Notebook
# 
# This notebook contains the model training pipeline used for EEG classification. An overview of this notebook is as follows
# 
# 1. Training/Testing dataset creation
# 2. Auto Encoder Model
# 3. LSTM Classification Model
# 4. Overall Results

# import some useful libraries

import os
import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# model creation
import torch.nn as nn
import torch.nn.functional as F
import torch

# attention module
from torchnlp.nn.attention import Attention

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
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
    
    dataset = np.vstack([np.expand_dims(x,0) for x in dataset]) # (N, window_size, 66)
    print(f"Time + Feel trace + Channel set shape (N, window_size, 66):  {dataset.shape}")
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

# helper function
def encoder_split(features, labels, k_train_indices):
    '''
    further split each test a single k group into autoencoder/lstm training 
    maintain the same global label distribution 
    '''
    train_features = features[k_train_indices]
    lstm_train_features, encoder_train_features, lstm_train_labels, _  = train_test_split(train_features, labels[k_train_indices], test_size=0.2, stratify=labels[k_train_indices])
    return lstm_train_features, encoder_train_features, lstm_train_labels

# Models & Loader function
# Below we define the autoencoder and lstm model
class autoencoder(nn.Module):
    def __init__(self, num_features=12):
        super(autoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Dropout(p=0.7),
            nn.Linear(128, num_features))
        self.decoder = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.Sigmoid())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def encode(self, x):
        x = self.encoder(x)
        return x
    
    def decode(self, x):
        x = self.decoder(x)
        return x

# autoencoder model
# input: (N, 64)
# latent features: Z
# encoder: (N,64) -> (N,16) -> (N, Z)
# decoder: (N,Z) -> (N,16) -> (N, 64)

class lstm_classifier(nn.Module):
    def __init__(self, num_features=12, num_hidden=32, dropout=0.2, n_labels=5):
        super(lstm_classifier, self).__init__()
        
        self.hidden_size = num_hidden
        self.input_size = num_features
        self.n_classes = n_labels
        self.attn = Attention(self.hidden_size)
        
        self.lstm_1 = nn.LSTM(
            input_size =  self.input_size,
            hidden_size = self.hidden_size,
            num_layers = 1,
            batch_first=True
        )
        
        
        self.classify = nn.Sequential(
            nn.Linear(self.hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, self.n_classes))

    
    def forward(self,x):
        x, (h_t, c_t) = self.lstm_1(x)

        # x -> (N, seq_len, hidden_size)
        # h -> (1, N, hidden_size)
        x = self.classify(h_t.squeeze(0)) 
        return x

class classifier_dataset(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        'Initialization'
        self.x = features # (N, window_size, encoding)
        self.labels = labels # (N, 1)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.x)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        x = torch.from_numpy(self.x[index]).float() # eeg channels, lstm
        y = torch.from_numpy(np.array(self.labels[index])).long() # feel trace labels int value [0,n_labels]
        return x, y

class autoencoder_dataset(torch.utils.data.Dataset):
    def __init__(self, features):
        'Initialization'
        self.x = features # (N, window_size, 64)
        self.x = self.x.reshape(-1, 64) # (N*window_size, 64)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.x)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        x = torch.from_numpy(self.x[index]).float()
        y = x

        return x, y

# Training Loops
def train_encoder(model, num_epochs, batch_size, learning_rate, train_split):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs*3, eta_min=1e-8)

    criterion = nn.MSELoss()
    
    train_dataset = autoencoder_dataset(train_split)
        
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size=batch_size,
                                               num_workers=8,
                                               shuffle=True)
    
    train_metrics = []
    for epoch in range(num_epochs):
        
        # reset metrics
        cur_train_loss = 0 # loss
        cur_train_sim = 0 # cosine similarity
        
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

            # metrics
            cur_train_loss += loss.detach().cpu()
            cur_train_sim += cosine_similarity(y.detach().cpu().numpy(), y_hat.detach().cpu().numpy()).diagonal().mean()
            scheduler.step()
        
        # average metrics over loop
        train_loop_size = len(train_loader)
        cur_train_loss = cur_train_loss/train_loop_size
        cur_train_sim = cur_train_sim/train_loop_size
        
        
        train_metrics.append([cur_train_loss, cur_train_sim])
        
        # print(f'Epoch:{epoch+1},'\
        #       f'\nTrain Loss:{cur_train_loss},'\
        #       f'\nTrain Cosine Similarity:{cur_train_sim}')
        
    return train_metrics

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

# Run Training for encoder and classifier (with encoding) and test classifier
def encode_classifier_data(encoder_model, classifier_features, num_features=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder_model.eval()
    with torch.no_grad():
        prev_shape = classifier_features.shape # (N, window_size, 64)
        x = torch.from_numpy(classifier_features).float().reshape(-1,1,64).to(device) # encode the 64 channels
        x_encoded = encoder_model.encode(x).reshape(prev_shape[0], prev_shape[1], num_features).detach().cpu().numpy()
        return x_encoded
        

def main_runner(subject_choice=1, label_type='angle'):

    # Before we create the models, we will first prepare the data by splitting it and preprocessing it.
    eeg_feeltrace_dir = 'ALIGNED_DATA' # directory containing *.csv files
    # hyper parameters
    window_size = 500 # must be an int in milliseconds
    subject_num = subject_choice # which subject to choose [1-16]
    k_fold = 5 # k for k fold validation
    #label_type = 'angle' # 'angle' or 'pos' or 'both'
    num_classes = 3 if label_type != 'both' else 9 # number of classes to discretize the labels into
    num_features = 16 # latent space features for encoder
    encoder_learning_rate = 1e-3
    classifier_learning_rate = 1e-3 # adam learning rate
    encoder_train_epochs = 30 # train encoder duration
    classifier_train_epochs = 30 # train classifier duration
    classifier_hidden = 128 # LSTM parameter, the larger the more complicated the model

    dataset, labels, indices = load_and_split_dataset(eeg_ft_dir = 'ALIGNED_DATA', split_size=window_size, subject_num = subject_num, k=k_fold, label_type=label_type, num_classes=num_classes)
    print(f"Label class bincount: {np.bincount(labels, minlength=num_classes)}")
    
    k_acc = [] # accuracies for each fold
    k_f1 = [] # f1 score for each fold
    k_cm = [] # confusion matrix for each fold

    for cur_k in range(len(indices)):
        print(f"Training k={cur_k}")
        train_index, test_index = indices[cur_k]

        lstm_train_features, encoder_train_features, lstm_train_labels = encoder_split(dataset, labels, train_index)


        encoder_model = autoencoder(num_features=num_features)
        classifier_model = lstm_classifier(num_features=num_features, num_hidden=classifier_hidden, dropout=0.5, n_labels=num_classes)

        print('Training Encoder!')
        encoder_train_metrics = train_encoder(encoder_model, encoder_train_epochs, batch_size=2048, learning_rate=encoder_learning_rate, train_split=encoder_train_features[:,:,2:])

        encoded_train_features = encode_classifier_data(encoder_model, lstm_train_features[:,:,2:], num_features=num_features)
        print('Training Classifier!')
        classifier_train_metrics = train_classifier(classifier_model, classifier_train_epochs, batch_size=128, learning_rate=classifier_learning_rate, features=encoded_train_features, labels=lstm_train_labels, num_classes=num_classes)


        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lstm_test_features, lstm_test_labels =  dataset[test_index], labels[test_index]
        print(f"Test label distribution: {np.bincount(lstm_test_labels, minlength=num_classes)}")
        encoded_test_features = encode_classifier_data(encoder_model, lstm_test_features[:,:,2:], num_features=num_features)

        with torch.no_grad():
            classifier_model.eval()
            x_encoded  = torch.from_numpy(encoded_test_features).float().to(device)
            y = lstm_test_labels
            y_hat = classifier_model(x_encoded)
            y_hat = F.softmax(y_hat.detach(), dim=-1).cpu().numpy()
            preds = y_hat

        # fig, axs = plt.subplots(figsize=(20,20), dpi=120)
        # axs.set_title(f"LSTM Feel Trace Model Confusion Matrix - Emotion-as-{label_type}", fontsize=20)
        # axs.set_xlabel("Predicted Label", fontsize=15)
        # axs.set_ylabel("True Label", fontsize=15)


        prf = precision_recall_fscore_support(lstm_test_labels, np.array([x.argmax() for x in preds]), average='macro', zero_division=0)
        acc = np.mean(lstm_test_labels == np.array([x.argmax() for x in preds]))
        print(f"Precision: {prf[0]}")
        print(f"Recall: {prf[1]}")
        print(f"F1-Score: {prf[2]}")
        print(f"Accuracy: {acc}")

        k_acc.append(acc)
        k_f1.append(prf[2])

        cm = confusion_matrix(lstm_test_labels, [x.argmax() for x in preds], labels=np.arange(num_classes), normalize=None)
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

    result_df.to_csv('eeg_classification_result_both.csv', index=False)


if __name__ == '__main__':
    run()
