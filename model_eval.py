#!/usr/bin/env python
# coding: utf-8

# In[1]:


import fastbook
fastbook.setup_book()


# In[2]:


import os
from fastai.vision.all import *
import torch
from fastai.layers import Lambda
from fastai.tabular.all import *
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import ElasticNet
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import socket
from sklearn.preprocessing import RobustScaler
from torch.utils.data import WeightedRandomSampler



# In[3]:


# globlas
#print computer name:
comp_name = str(socket.gethostname())
base_path = str('')
if (comp_name == 'shairlab.upper.2080'):
    base_dir = Path('/home/jesse/lab/trans_stamp/')


#function for running analysis
def regression_baseline(genexp, y_col, non_score_cols):
    '''

    :param genexp: a df of genexp
    :param y_col: the name of the label col
    :param non_score_cols: name of cols to drop for X as a list
    :return: nome
    '''


    # Preprocess the data (Assuming 'label' is your target variable)
    X = genexp .drop(non_score_cols, axis=1)
    y = genexp [y_col]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    # Train Random Forest
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)

    # Predictions and Evaluation for Random Forest
    rf_predictions = rf.predict(X_test)
    print('random forests:\n',classification_report(y_test, rf_predictions))


    # Train Elastic Net
    en = ElasticNet()
    en.fit(X_train, y_train)

    # Predictions and Evaluation for Elastic Net
    # Note: Elastic Net is a regression model; you might need to convert its predictions to a binary format for classification
    en_predictions = en.predict(X_test)
    en_predictions = [1 if p > 0.5 else 0 for p in en_predictions]  # Example threshold
    print('elsatic net:\n',classification_report(y_test, en_predictions))



# In[6]:


def create_sample_weights(target_data):
    class_sample_counts = target_data.value_counts()
    weight = 1. / class_sample_counts
    samples_weight = weight[target_data].values
    return torch.DoubleTensor(samples_weight)


# In[7]:


def create_dls(data,scale=True,scale_cols =2, regression = False, weighted = True):
    
    #apply scaling
    if scale:
        scaler = RobustScaler()
        data[data.columns[scale_cols:]] = scaler.fit_transform(data[data.columns[scale_cols:]])

    #get name of cols
    #find col with 'labal'
    label_col = [col for col in data.columns if 'label' in col]
    #print warning if more than one label col
    if len(label_col) > 1:
        print('more than one label col')
        print(label_col)
        print('using first label col')
        label_col = label_col[0]
    else:
        label_col = label_col[0]

    #get metadata cols: cols with label or sample_id
    metadata_cols = [col for col in data.columns if 'label' in col or 'Sample' in col]

    #print regression baseline:
    if regression:
        regression_baseline(data, label_col, metadata_cols)

    
    #create data loaders

    # Preprocess the data (Assuming 'label' is your target variable)
    #split scgpt_df into val and train
    train_df = data.sample(frac=0.8,random_state=200) #random state is a seed value
    val_df = data.drop(train_df.index)

    # create stacked tensor of sample vectors. each row is a sample, each col is a dim
    train_x = train_df.drop(metadata_cols, axis=1)
    train_x = torch.stack([tensor(row) for row in train_x.values], dim=0)
    train_y = tensor(train_df[label_col].values).float()
    
    val_x = val_df.drop(metadata_cols, axis=1)
    val_x = torch.stack([tensor(row) for row in val_x.values], dim =0)
    val_y = tensor(val_df[label_col].values).float()

    #create data sets from train and val x and y
    train_dset = list(zip(train_x,train_y))
    val_dset = list(zip(val_x,val_y))
    
    #create weighted sampler
    if weighted:
        train_sample_weights = create_sample_weights(train_df[label_col])
        train_sampler = WeightedRandomSampler(train_sample_weights, len(train_sample_weights))
        dl_train = DataLoader(train_dset, batch_size=1, sampler=train_sampler)
    #if not weighted, create dl without sampler
    if not weighted:
        dl_train = DataLoader(train_dset, batch_size=1)
    
    #create data loaders
    dl_val = DataLoader(val_dset, batch_size=1)

    #return dls
    return (dl_train, dl_val)


# 

# # create a simple neural net
# 
# now that we've fixed the gradiants and have proper learning

# In[8]:


def stamp_loss(preds, targs):
    preds = preds.sigmoid()
    return torch.where(targs==1, 1-preds, preds).mean()


# In[9]:


def batch_accuracy(xb, yb):
    preds = xb.sigmoid()
    correct = (preds>0.5) == yb
    return correct.float().mean()


# In[10]:


def first_nn(input_dim):
    nn_model = nn.Sequential(
    nn.Linear(input_dim, 50),
    nn.ReLU(),
    nn.Linear(50, 50),
    nn.ReLU(),
    nn.Linear(50, 1),
    Lambda(lambda x: x.squeeze(-1))
    )
    return nn_model


# In[11]:


def run_nn(dl_train, dl_val, input_dim, epochs = 10, lr =0.1, find_lr = False, report = True):
    #having creted dataloadrs from data sets for train and cal, create a dls 
    dls = DataLoaders(dl_train, dl_val)

    model = first_nn(input_dim)
    
    #optimisers: Adam, AdamW, RMSProp, SGD
    #loss:nn.CrossEntropyLoss, nn.BCEWithLogitsLoss,
    # nn.CrossEntropyLoss(weight=class_weights)
    
    # Create your learner
    learn = Learner(dls, model, opt_func = Adam, loss_func=BCEWithLogitsLossFlat, metrics=error_rate)

    #find suggested lr
    if find_lr:
        lr_suggestion = learn.lr_find()
        # Accessing the suggested learning rate
        lr = lr_suggestion.lr_min
    #train
    learn.fit(epochs)

    # create classification report
    if report:
        '''# Get predictions
        preds, targets = learn.get_preds(dl=dls.valid)
        predictions = preds.argmax(dim=1)
    
        # Convert tensors to numpy arrays for compatibility with scikit-learn
        predictions_np = predictions.numpy()
        targets_np = targets.numpy()
    
        # Generate classification report
        print('Neural Network:\n', classification_report(targets_np, predictions_np))'''
        interp = ClassificationInterpretation.from_learner(learn)
        interp.plot_confusion_matrix()
        plt.show()
        # Get top losses
        top_losses = interp.top_losses()
        

data_path =Path(base_dir / 'scgpt/data/scgpt_embeddings/brca_sub_scgpt_emb.csv') 
scgpt_df = pd.read_csv(data_path)


# In[14]:


# load raw genexp data
raw_genxp_path = Path(base_dir / 'scgpt/data/bulk_brca_erbb2/brca_ERBB2_genexp_oncosig_labels_gene_map.csv') 
raw_genexp = pd.read_csv(raw_genxp_path)


# In[15]:


data = raw_genexp

#get dls
dl_train, dl_val = create_dls(data, weighted = False)


# In[16]:


#trouble shoot dim problem with training

run_nn(dl_train, dl_val, input_dim=855, epochs=10, lr=0.1, find_lr=False, report = False)
