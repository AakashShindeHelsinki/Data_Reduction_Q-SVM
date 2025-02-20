from sklearn.datasets import make_blobs, make_classification, make_gaussian_quantiles
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import csv
import shortuuid
import numpy as np
from synthetic_data.synthetic_data import make_tabular_data

import tensorflow as tf
from tensorflow.keras.models import Model 
from tensorflow.keras import layers, losses


def data_generator_sklearn():
    DataID = shortuuid.uuid()
    DataGenAlgo = ' make_classification'
    Sample_size = 2000
    nclass = 2
    GenRandomState = 1
    nInfo = 10
    nFeature = 16
    nRedun = 4
    nRep = 2
    nCluster = 1
    classSep = 2


    X, y = make_classification(n_samples=Sample_size, n_classes=nclass, n_features=nFeature, n_redundant=nRedun, n_repeated=nRep,
                               n_informative=nInfo, n_clusters_per_class=nCluster, random_state=GenRandomState, class_sep=classSep)

    data_file = {'DataID':DataID,'DataGenAlgo':DataGenAlgo,'Sample_size':Sample_size,'nclass':nclass,
                     'GenRandomState':GenRandomState,'nInfo':nInfo,'nFeature':nFeature,'nRedun':nRedun,
                     'nRep':nRep,'nCluster':nCluster,'classSep':classSep}


    return X,y, DataID, data_file



def data_generator_synthetic_data_capital1():

    DataID = shortuuid.uuid()
    DataGenAlgo = 'synthetic_data'
    Sample_size = 2000
    nclass = 2
    GenRandomState = 1
    nInfo = 10
    nFeature = 16
    nRedun = 2
    nNusiance = 4
    pthresh = 0.75
    expr2 ="x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10"
    col_map = {"x1": 1, "x2": 0,"x3":1, "x4": 1, "x5": 0, "x6": 1, "x7": 1, "x8": 0,"x9":1, "x10": 0}
    cov = np.array([[1. , 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
       [0.8, 1. , 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
       [0.8, 0.8, 1. , 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
       [0.8, 0.8, 0.8, 1. , 0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
       [0.8, 0.8, 0.8, 0.8, 1. , 0.8, 0.8, 0.8, 0.8, 0.8],
       [0.8, 0.8, 0.8, 0.8, 0.8, 1. , 0.8, 0.8, 0.8, 0.8],
       [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 1. , 0.8, 0.8, 0.8],
       [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 1. , 0.8, 0.8],
       [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 1. , 0.8],
       [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 1. ]])

    X, y_reg, y_prob, y_label = make_tabular_data(n_samples=Sample_size, n_informative=nInfo, n_redundant=nRedun ,n_nuisance=nNusiance, cov=cov, n_classes=nclass, col_map=col_map, expr=expr2, p_thresh=pthresh, seed=GenRandomState)

    data_file = {'DataID':DataID,'DataGenAlgo':DataGenAlgo,'Sample_size':Sample_size,'nclass':nclass,
                     'GenRandomState':GenRandomState,'nInfo':nInfo,'nFeature':nFeature,'nRedun':nRedun,
                     'nNusiance':nNusiance,'expression': expr2,'pthreshold':pthresh, 'Col-Map': col_map,
                     'covariance matrix':cov, 'y regression': y_reg, 'y probabilities': y_prob}

    y = np.round(y_prob).astype(int)
    return X,y, DataID, data_file


def pca_algo(q_num, X_train, X_test):
    pca = PCA(q_num)
    X_train = pca.fit_transform(X_train)
    X_test = pca.fit_transform(X_test)

    return X_train,X_test


class Autoencoder(Model):
    def __init__(self, latent_dim, shape):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.shape = shape
        self.encoder = tf.keras.Sequential([
            layers.Input(shape = shape),
            layers.Flatten(),
            layers.Dense(32, activation = 'sigmoid'),
            layers.Dense(16, activation = 'relu'),
            layers.Dense(64, activation = 'relu'),
            layers.Dense(32, activation = 'sigmoid'),
            layers.Dense(latent_dim, activation='relu'),
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(tf.math.reduce_prod(shape).numpy(), activation='sigmoid'),
            layers.Reshape(shape)
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded



def autoencoder_algo(q_num, X_train, X_test):
    shape = X_test.shape[1:]
    autoencoder = Autoencoder(latent_dim=q_num, shape=shape)
    #autoencoder.compile(optimizer = 'adam', loss= losses.BinaryCrossentropy())
    autoencoder.compile(optimizer = 'adam', loss = losses.MeanSquaredError())
    autoencoder.fit(X_train,X_train, epochs = 80, shuffle = True, validation_data = (X_test,X_test))

    X_train, X_test = autoencoder.encoder(X_train).numpy(), autoencoder.encoder(X_test).numpy()

    return X_train, X_test


def data_load_and_process(q_num, data_gen, data_redu = 'no_redu' ):
    traintestSplit = 0.2
    Shuf = False
    SplitRandomState = 42

    field_names = ['DataID','DataGenAlgo','Sample_size','nclass','GenRandomState','nInfo','nFeature','nRedun','nRep','nCluster','classSep',
                   'nNusiance','expression','pthreshold', 'Col-Map', 'covariance matrix', 'y regression', 'y probabilities',
                   'train-testSplit','Shuf','SplitRandomState','X_train','Y_train','X_test','Y_test']

    if data_gen == 'sklearn_make_class':
        X, y, DataID, data_file = data_generator_sklearn()
    elif data_gen == 'capital1_synthetic_data':
        X, y, DataID, data_file = data_generator_synthetic_data_capital1()


    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = traintestSplit, shuffle=Shuf, random_state=SplitRandomState)

    
    if data_redu == 'pca':
        X_train, X_test = pca_algo(q_num, X_train, X_test)
    elif data_redu == 'autoencode':
        X_train, X_test = autoencoder_algo(q_num, X_train, X_test)

    
    update_vals = {'X_train' : X_train,'X_test' : X_test,'Y_train' : Y_train, 'Y_test' : Y_test,
                   'train-testSplit':traintestSplit, 'Shuf':Shuf,'SplitRandomState':SplitRandomState}


    data_file.update(update_vals)
        

    with open('Results/TESTFILE_Train_Binary_Synthetic.csv', 'a') as csvfile: 
        writer = csv.DictWriter(csvfile, fieldnames = field_names)
        writer.writeheader() 
        writer.writerows([data_file])

    return X_train, Y_train, X_test, Y_test, DataID




