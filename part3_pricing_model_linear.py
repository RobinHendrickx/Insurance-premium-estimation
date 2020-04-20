from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
import pickle
import numpy as np


import matplotlib.pyplot as plt
import copy
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from sklearn import preprocessing
from scipy import stats
import copy

from part2_claim_classifier import EarlyStopping, Insurance_NN, ClaimClassifier, \
    WeightedBCELoss, analyse, weighted_binary_cross_entropy

import torch
import torch.nn as nn

def fit_and_calibrate_classifier(classifier, X, y):
    # DO NOT ALTER THIS FUNCTION
    X_train, X_cal, y_train, y_cal = train_test_split(
        X, y, train_size=0.85, random_state=0)
    classifier = classifier.fit(X_train, y_train)

    # This line does the calibration for you
    y_cal = y_cal.to_numpy()[:, 0]
    calibrated_classifier = CalibratedClassifierCV(
        classifier, method='sigmoid', cv='prefit').fit(X_cal, y_cal)
    return calibrated_classifier


class Insurance_NN_4(nn.Module):
    def __init__(self):
        super(Insurance_NN_4, self).__init__()

        self.apply_layers = nn.Sequential(
            # 2 fully connected hidden layers of 8 neurons goes to 1
            # 9 - 1
            nn.Linear(9, 1),
            nn.Sigmoid()
        )

    # Defining the forward pass
    def forward(self, x):
        x = self.apply_layers(x)
        return x.view(len(x))


# class for part 3
class PricingModelLinear():
    # YOU ARE ALLOWED TO ADD MORE ARGUMENTS AS NECESSARY
    def __init__(self, calibrate_probabilities=False):
        """
        Feel free to alter this as you wish, adding instance variables as
        necessary.
        """
        self.y_mean = None
        self.calibrate = calibrate_probabilities
        # =============================================================
        # READ ONLY IF WANTING TO CALIBRATE
        # Place your base classifier here
        # NOTE: The base estimator must have:
        #    1. A .fit method that takes two arguments, X, y
        #    2. Either a .predict_proba method or a decision
        #       function method that returns classification scores
        #
        # Note that almost every classifier you can find has both.
        # If the one you wish to use does not then speak to one of the TAs
        #
        # If you wish to use the classifier in part 2, you will need
        # to implement a predict_proba for it before use
        # =============================================================
        self.base_classifier = ClaimClassifier(Insurance_NN_4()) # ADD YOUR BASE CLASSIFIER HERE


    # YOU ARE ALLOWED TO ADD MORE ARGUMENTS AS NECESSARY TO THE _preprocessor METHOD
    def _preprocessor(self, X_raw):
        """Data preprocessing function.

        This function prepares the features of the data for training,
        evaluation, and prediction.

        Parameters
        ----------
        X_raw : ndarray
            An array, this is the raw data as downloaded

        Returns
        -------
        X: ndarray
            A clean data set that is used for training and prediction.
        """
        # =============================================================
        # YOUR CODE HERE
        X_raw = copy.deepcopy(
            X_raw[['pol_coverage', 'vh_age', 'vh_din', 'vh_fuel',
                   'vh_sale_begin', 'vh_sale_end', 'vh_speed', 'vh_value',
                   'vh_weight']])

        X_raw.dropna(how="any", inplace=True)
        X_raw = self.integer_encode(X_raw)

        if not isinstance(X_raw, np.ndarray):
            X_raw = X_raw.to_numpy(dtype=np.float)

        min_max_scaler = preprocessing.MinMaxScaler()
        X_raw = min_max_scaler.fit_transform(X_raw)

        return X_raw.astype(np.float32)

    def fit(self, X_raw, y_raw, claims_raw, prepro = True):
        """Classifier training function.

        Here you will use the fit function for your classifier.

        Parameters
        ----------
        X_raw : ndarray
            This is the raw data as downloaded
        y_raw : ndarray
            A one dimensional array, this is the binary target variable
        claims_raw: ndarray
            A one dimensional array which records the severity of claims

        Returns
        -------
        self: (optional)
            an instance of the fitted model

        """
        nnz = np.where(claims_raw != 0)[0]
        self.y_mean = np.mean(claims_raw[nnz])
        # =============================================================
        # REMEMBER TO A SIMILAR LINE TO THE FOLLOWING SOMEWHERE IN THE CODE
        if prepro:
            X_clean = self._preprocessor(X_raw)
        else:
            X_clean = X_raw

        # THE FOLLOWING GETS CALLED IF YOU WISH TO CALIBRATE YOUR PROBABILITES
        if self.calibrate:
            self.base_classifier = fit_and_calibrate_classifier(
                self.base_classifier, X_clean, y_raw)
            self.save_model()
        else:
            self.base_classifier = self.base_classifier.fit(X_clean, y_raw)
            self.save_model()
        return self.base_classifier

    def predict_claim_probability(self, X_raw):
        """Classifier probability prediction function.

        Here you will implement the predict function for your classifier.

        Parameters
        ----------
        X_raw : ndarray
            This is the raw data as downloaded

        Returns
        -------
        ndarray
            A one dimensional array of the same length as the input with
            values corresponding to the probability of beloning to the
            POSITIVE class (that had accidents)
        """
        # =============================================================
        # REMEMBER TO A SIMILAR LINE TO THE FOLLOWING SOMEWHERE IN THE CODE
        X_clean = self._preprocessor(X_raw)
        #X_clean = X_raw
        # return probabilities for the positive class (label 1)
        return  self.base_classifier.predict_proba(X_clean)[:,1]

    def predict_premium(self, X_raw):
        """Predicts premiums based on the pricing model.

        Here you will implement the predict function for your classifier.

        Parameters
        ----------
        X_raw : numpy.ndarray
            A numpy array, this is the raw data as downloaded

        Returns
        -------
        numpy.ndarray
            A one dimensional array of the same length as the input with
            values corresponding to the probability of beloning to the
            POSITIVE class (that had accidents)
        """
        # =============================================================
        # REMEMBER TO INCLUDE ANY PRICING STRATEGY HERE.
        # For example you could scale all your prices down by a factor

        return self.predict_claim_probability(X_raw) * self.y_mean * 0.2725

    def save_model(self):
        """Saves the class instance as a pickle file."""
        # =============================================================
        with open('part3_pricing_model_linear.pickle', 'wb') as target:
            pickle.dump(self, target)


    def load_data(self, filename):
        """
        Function to load data from file
        Args:
            filename (str) - name of .txt file you are loading data from
        Output:
            (x, y) (tuple) - x: 2D array of training data where each row
            corresponds to a different sample and each column corresponds to a
            different attribute.
                            y: 1D array where each index corresponds to the
            ground truth label of the sample x[index][]
        """

        dat = pd.read_csv("part3_training_data.csv")
        #dat.drop(columns=["drv_sex2"], inplace=True)
        #dat.dropna(how="any", inplace=True)
        x = dat.drop(columns=["claim_amount", "made_claim"])
        y = dat["made_claim"]
        y1 = dat["claim_amount"]
        y2 = y1[y1!=0]

        return x, y, y2.to_numpy(), y1

    def separate_pos_neg(self, x, y):

        # Separate into positive and negative samples
        pos_train_y = []
        pos_train_x = np.empty((0, x.shape[1]), np.float32)
        neg_train_y = []
        neg_train_x = np.empty((0, x.shape[1]), np.float32)
        for i in range(y.shape[0]):
            if y[i] == 1:
                pos_train_y.append(y[i])
                pos_train_x = np.vstack((pos_train_x, x[i]))
            else:
                neg_train_y.append(y[i])
                neg_train_x = np.vstack((neg_train_x, x[i]))

        neg_train_y = np.array(neg_train_y, dtype=np.float32)
        pos_train_y = np.array(pos_train_y, dtype=np.float32)

        return (neg_train_x, neg_train_y), (pos_train_x, pos_train_y)

    def integer_encode(self, x):
        """
        Encode all columns containing strings with unique numbers for every
        category type
        """
        x = x.to_numpy(dtype=str)
        for att_i in range(x.shape[1]):
            try:
                float(x[0, att_i])

            except ValueError:
                values = x[:, att_i]
                # integer encode
                label_encoder = LabelEncoder()
                integer_encoded = label_encoder.fit_transform(values)
                x[:, att_i] = integer_encoded
        return x.astype(float)

def load_model():
    # Please alter this section so that it works in tandem with the save_model method of your class
    with open('part3_pricing_model_linear.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    return trained_model



if __name__ == "__main__":
    test = load_model()
    x, y, claims_raw, y1 = test.load_data("part3_training_data.csv")
    print(test.predict_claim_probability(x).shape)
    test.base_classifier.evaluate_architecture(True)
    """
    test = PricingModelLinear()
    x, y, claims_raw, y1 = test.load_data("part3_training_data.csv")

    print(x.shape, y.shape, claims_raw.shape)
    x = test._preprocessor(x)
    train_data, test_data = test.base_classifier.separate_data(x, y, y1)

    print(train_data[0].shape, train_data[1].shape)

    nnz = np.where(claims_raw != 0)[0]
    print(claims_raw[nnz])
    y_mean = np.mean(claims_raw[nnz])
    y_std = np.std(claims_raw[nnz])

    print(y_mean, y_std)
    #test2 = load_model()
    #print(test2.predict_premium(x))
    test.fit(train_data[0], train_data[1], claims_raw, False)
    #test.predict_claim_probability(test_data[0])
    test.base_classifier.evaluate_architecture(True)
    """
