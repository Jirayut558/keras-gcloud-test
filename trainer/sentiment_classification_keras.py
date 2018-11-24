import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
import re
import glob
from keras import callbacks
from tensorflow.python.lib.io import file_io
import argparse


def Model(max_feature,embedding_dim,lstm_unit,input_dim):

    model = Sequential()
    model.add(Embedding(max_feature, embedding_dim, input_length=input_dim))
    model.add(SpatialDropout1D(0.4))
    model.add(LSTM(lstm_unit, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model

def preparing_data(data,max_fatures=3000):
    data = data[['CONTENT', 'CLASS']]
    data['CONTENT'] = data['CONTENT'].apply(lambda x: x.lower())
    data['CONTENT'] = data['CONTENT'].apply((lambda x: re.sub('[^a-zA-z0-9\s]', '', x)))

    tokenizer = Tokenizer(num_words=max_fatures, split=' ')
    tokenizer.fit_on_texts(data['CONTENT'].values)
    X = tokenizer.texts_to_sequences(data['CONTENT'].values)
    X = pad_sequences(X)
    Y = pd.get_dummies(data['CLASS']).values

    return X,Y

def read_multiple_csv(path):
    with file_io.FileIO(path, mode='r') as file:
        df = pd.read_csv(file, index_col=None, header=0)
    return df

def main(job_dir,**args):
    #tuning parameter
    max_feature = 3000
    embedding_dim = 128
    lstm_unit = 128
    batch_size = 32
    epoch = 7

    csv_folder = job_dir + 'data/Youtube01-Psy.csv'
    logs_path = job_dir + 'logs/tensorboard'

    #read data set from csv folder
    df = read_multiple_csv(csv_folder)

    #preparing data for training data
    X,Y = preparing_data(df,max_feature)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

    ## Adding the callback for TensorBoard and History
    tensorboard = callbacks.TensorBoard(log_dir=logs_path, histogram_freq=0, write_graph=True, write_images=True)

    #Initial model
    model = Model(max_feature,embedding_dim,lstm_unit,X.shape[1])

    ##fitting the model
    model.fit(x=X_train, y=Y_train, epochs=epoch, verbose=1, batch_size=batch_size, callbacks=[tensorboard],validation_data = (X_test, Y_test))

    # Save model.h5 on to google storage
    model.save('model.h5')
    with file_io.FileIO('model.h5', mode='rb') as input_f:
        with file_io.FileIO(job_dir + 'models/model.h5', mode='w+') as output_f:
            output_f.write(input_f.read())

##Running the app
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Input Arguments
    parser.add_argument(
      '--job-dir',
      help='GCS location to write checkpoints and export models',
      required=True
    )
    args = parser.parse_args()
    arguments = args.__dict__

    main(**arguments)