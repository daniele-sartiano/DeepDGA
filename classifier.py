#!/usr/bin/env python

import sys
import time
import argparse

import numpy as np

from keras.models import Model, load_model
from keras.layers import Dense, Activation, Embedding, LSTM, Input, Bidirectional
from keras.optimizers import RMSprop, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import np_utils

from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler

class Reader(object):
    DIM = 255
    def __init__(self, input):
        self.input = input

    def read(self):
        X,  y = [], []
        for example in self.input:
            domain, label = example.strip().rsplit(',',1)
            X.append(self.convert(domain))
            y.append(self.get_label(label))
        return np.asarray(X), np.asarray(y)

    @staticmethod
    def convert(domain):
        d = np.zeros(255)
        for i, ch in enumerate(domain):
            d[i] = ord(ch)
        return d

    def get_label(self, label):
        return int(label)
    
    def convert_labels(self, labels):
        return labels

    @staticmethod
    def revert(array):
        return ''.join([chr(int(el)) for el in array if el != 0])

class MultiClassReader(Reader):

    def __init__(self, input, nb_classes):
        self.input = input
        self.nb_classes = nb_classes
    
    def get_label(self, label):
        print(np_utils.to_categorical(label, self.nb_classes))
        return np_utils.to_categorical(label, self.nb_classes)

    def convert_labels(self, labels):
        if self.nb_classes > 1:
            return np.argmax(labels, axis=-1)
        return labels
    
class Classifier(object):
    LEN_CHARS = 255

    def __init__(self, file_model='', nb_classes=1, embeddings_dim=50, batch_size=1024, epochs=20, lstm_size=1024, bidirectional=False, dropout=0.2, nodense=False):
        self.file_model = file_model
        self.nb_classes = nb_classes
        self.embeddings_dim = embeddings_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.lstm_size = lstm_size
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.nodense= nodense
        self._model()

    def _model(self):
        domain_input = Input(shape=(Reader.DIM,))
        embeddings = Embedding(
            input_dim=self.LEN_CHARS,
            output_dim=self.embeddings_dim,
            input_length=Reader.DIM,
            mask_zero=True
        )(domain_input)
        if self.bidirectional:
            lstm = Bidirectional(LSTM(self.lstm_size, dropout=self.dropout, recurrent_dropout=self.dropout))(embeddings)
        else:
            lstm = LSTM(self.lstm_size, dropout=self.dropout, recurrent_dropout=self.dropout)(embeddings)
        if not self.nodense:
            lstm = Dense(256, activation='relu')(lstm)
        out = Dense(1, activation='sigmoid')(lstm)
        self.model = Model(inputs=[domain_input], outputs=out)
        self.model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])
        self.model.summary()

    def save_model(self):
        self.model.save(self.file_model)

    def load_model(self):
        self.model = load_model(self.file_model)
        
    def train(self, X, y):
        self.model.fit(X, y,
                       shuffle=True,
                       batch_size=self.batch_size,
                       epochs=self.epochs,
                       callbacks=[
                           EarlyStopping(verbose=True, patience=5, monitor='val_loss'),
                           ModelCheckpoint(self.file_model + '.tmp', monitor='val_loss', verbose=True, save_best_only=True)
                       ]
        )

    def evaluate(self, X_dev, y_dev):
        out = self.model.evaluate(X_dev, y_dev, batch_size=1024)
        print('Test score:', out)

    def predict(self, X_test, verbose=1):
        y = self.model.predict(X_test, verbose=verbose)
        y = (y > 0.5).astype('int32')
        return y

    def classification_report(self, y_gold, y_pred):
        return classification_report(y_gold, y_pred)

    def confusion_matrix(self, y_gold, y_pred):
        return confusion_matrix(y_gold, y_pred)


class BotNetClassifier(Classifier):

    def _model(self):
        domain_input = Input(shape=(Reader.DIM,))
        embeddings = Embedding(
            input_dim=self.LEN_CHARS,
            output_dim=self.embeddings_dim,
            input_length=Reader.DIM,
            mask_zero=True
        )(domain_input)
        if self.bidirectional:
            lstm = Bidirectional(LSTM(self.lstm_size, dropout=self.dropout, recurrent_dropout=self.dropout))(embeddings)
        else:
            lstm = LSTM(self.lstm_size, dropout=self.dropout, recurrent_dropout=self.dropout)(embeddings)
        if not self.nodense:
            lstm = Dense(256, activation='relu')(lstm)
        out = Dense(self.nb_classes, activation='softmax')(lstm)
        self.model = Model(inputs=[domain_input], outputs=out)
        self.model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])
        self.model.summary()

    def predict(self, X_test, verbose=1):
        y = self.model.predict(X_test, verbose=verbose)
        y = y.argmax(axis=-1)
        return y

        
def main():
    parser = argparse.ArgumentParser(description='DGA classifier')
    subparsers = parser.add_subparsers()

    common_args = [
        (['-f', '--file-model'], {'help':'file model', 'type':str, 'default':'web.model'}),
        (['-i', '--input'], {'help':'input', 'help': 'input, default standard input', 'type':str, 'default':None})
    ]

    parser_train = subparsers.add_parser('train')
    parser_train.set_defaults(which='train')

    parser_train.add_argument('-e', '--epochs', help='Epochs', type=int, default=20)
    parser_train.add_argument('-nb', '--nb-classes', help='nb classes', type=int, default=1)
    parser_train.add_argument('-ed', '--embedding-dim', help='Embedding dim', type=int, default=50)
    parser_train.add_argument('-b', '--batch-size', help='Batch size', type=int, default=1024)
    parser_train.add_argument('-lstm', '--lstm-size', help='LSTM layer size', type=int, default=1024)
    parser_train.add_argument('-dropout', '--dropout', help='Dropout', type=float, default=0.2)
    parser_train.add_argument('-bi', '--bidirectional', help='Bidirectional LSTML', action='store_true')
    parser_train.add_argument('-nodense', '--nodense', help='Without dense layer after lstml', action='store_true')
    
    for arg in common_args:
        parser_train.add_argument(*arg[0], **arg[1])

    parser_predict = subparsers.add_parser('predict')
    parser_predict.set_defaults(which='predict')

    for arg in common_args:
        parser_predict.add_argument(*arg[0], **arg[1])

    args = parser.parse_args()

    input = open(args.input) if args.input else sys.stdin

    r = Reader(input) if args.nb_classes == 1 else MultiClassReader(input, args.nb_classes)
    
    if args.which == 'train':        
        X, y = r.read()
        X_train, X_dev, y_train, y_dev = train_test_split(X, y, random_state=42)

        name_file = '{}.model'.format(int(time.time()))

        params = {
            'file_model': args.file_model,
            'nb_classes': args.nb_classes,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lstm_size': args.lstm_size,
            'bidirectional': args.bidirectional,
            'dropout':args.dropout,
            'nodense':args.nodense
        }
        
        classifier = Classifier(**params) if args.nb_classes == 1 else BotNetClassifier(**params)
        
        classifier.train(X_train, y_train)
        classifier.save_model()
            
        classifier.evaluate(X_dev, y_dev)
        y_pred = classifier.predict(X_dev)
        classifier.model.summary()
        print(classifier.classification_report(r.convert_labels(y_dev), y_pred))

    elif args.which == 'predict':
        X, _ = r.read()
        classifier = Classifier(file_model=args.file_model)
        classifier.load_model()
        y_pred = classifier.predict(X)
        for i, el in enumerate(X):
            print(Reader.revert(el), y_pred[i])

if __name__ == '__main__':
    main()
