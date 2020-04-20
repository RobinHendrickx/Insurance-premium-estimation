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
    y_cal = y_cal.to_numpy()[:,0]
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

class Insurance_NN_3(nn.Module):
    def __init__(self):
        super(Insurance_NN_3, self).__init__()

        self.apply_layers = nn.Sequential(
            # 2 fully connected hidden layers of 8 neurons goes to 1
            # 9/6 - (100 - 10) - 1
            nn.Linear(9, 16),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(),
            #nn.Linear(70, 10),
            #nn.LeakyReLU(inplace=True),
            #nn.Dropout(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    # Defining the forward pass
    def forward(self, x):
        x = self.apply_layers(x)
        return x.view(len(x))

# class for part 3
class PricingModel():
    # YOU ARE ALLOWED TO ADD MORE ARGUMENTS AS NECESSARY
    def __init__(self, calibrate_probabilities=False):
        """
        Feel free to alter this as you wish, adding instance variables as
        necessary.
        """
        self.y_mean = None
        self.y_std = None
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
        self.base_classifier = ClaimClassifier(Insurance_NN_3()) # ADD YOUR BASE CLASSIFIER HERE


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
        X_raw = copy.deepcopy(X_raw[['pol_coverage', 'vh_age', 'vh_din', 'vh_fuel',
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
        self.y_std = np.std(claims_raw[nnz])
        print(self.y_mean, self.y_std)
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
            self.base_classifier.fit(X_clean, y_raw)
            self.save_model()
        return self

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

        # return probabilities for the positive class (label 1)
        return self.base_classifier.predict_proba(X_clean)[:,1]

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

        return self.predict_claim_probability(X_raw) * self.y_mean * 0.2775

    def save_model(self):
        """Saves the class instance as a pickle file."""
        # =============================================================
        with open('part3_pricing_model.pickle', 'wb') as target:
            pickle.dump(self, target)


    # -------- NEW FUNCTIONS -----------

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
        """
        # load data to single 2D array
        data_set = np.genfromtxt(filename, dtype=str, delimiter=',', skip_header=1)

        num_att = len(data_set[0])  # number of parameters

        x = np.array(data_set[:, :(num_att-2)], dtype=str)
        y = np.array(data_set[:, (num_att-1)], dtype=np.float)
        """

        return x, y, y2.to_numpy(), y1


    def set_axis_style(self, ax, labels):
        ax.get_xaxis().set_tick_params(direction='out')
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels)
        ax.set_xlim(0.25, len(labels) + 0.75)
        ax.set_xlabel('Sample name')

    def evaluate_input1(self, X_raw):
        """
        Function to evaluate data loaded from file

        """

        attributes = []
        for i in range(np.shape(X_raw)[1]):
            attributes.append(X_raw[:, i])


        fig, ax1 = plt.subplots(figsize=(18, 4))

        # type of plot
        ax1.boxplot(attributes)

        labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15,
                  16, 17, 18, 19, 20, 21, 22, 24, 24, 25, 26, 27, 28, 29, 30,
                  31, 32, 33, 34, 35]

        self.set_axis_style(ax1, labels)

        plt.subplots_adjust(bottom=0.15, wspace=0.05)
        # plt.show()
        plt.xlabel("Attribute Type")
        plt.ylabel("Attribute Value")

        plt.savefig("box_3.pdf", bbox_inches='tight')

        ####################

        plt.cla()
        ax1.violinplot(attributes)


        self.set_axis_style(ax1, labels)

        plt.subplots_adjust(bottom=0.15, wspace=0.05)
        plt.xlabel("Attribute Type")
        plt.ylabel("Attribute Value")


        plt.savefig("violin_3.pdf", bbox_inches='tight')

    def evaluate_input2(self, x, y):
        """
        Function to evaluate data loaded from file

        """

        # Separate positive and negative results

        (neg_x, neg_y), (pos_x, pos_y) = self.separate_pos_neg(x, y)
        attributes1 = []
        attributes2 = []
        for i in range(np.shape(neg_x)[1]):
            attributes1.append(neg_x[:, i])
            attributes2.append(pos_x[:, i])

        fig, axs = plt.subplots(2, figsize=(11, 11))

        # type of plot
        axs[0].boxplot(attributes1)
        axs[1].boxplot(attributes2)

        labels = np.genfromtxt("part3_training_data.csv", dtype=str,
                               delimiter=',',
                               max_rows=1)

        self.set_axis_style(axs[0], labels)
        self.set_axis_style(axs[1], labels)

        # plt.show()
        axs[0].set(xlabel="Attribute Type", ylabel="Attribute Value")
        axs[0].set_title("No Claim")
        axs[1].set(xlabel="Attribute Type", ylabel="Attribute Value")
        axs[1].set_title("Claim")

        plt.subplots_adjust(bottom=0.15, wspace=0.05)
        plt.savefig("compare_box_3.pdf", bbox_inches='tight')

    def evaluate_input3(self, x, y, split = 0):
        """
        Function to evaluate data loaded from file

        """

        # Separate positive and negative results
        if split == 0:
            (neg_x, neg_y), (pos_x, pos_y) = self.separate_pos_neg(x, y)
        else:
            (neg_x, neg_y), (pos_x, pos_y) = split
            print(split[0][0].shape, split[1][0].shape)

        attributes1 = []
        attributes2 = []
        difference = []
        difference2 = []
        for i in range(np.shape(neg_x)[1]):


            attributes1.append(np.mean(neg_x[:, i]))
            attributes2.append(np.mean(pos_x[:, i]))
            difference.append(((attributes2[i]-attributes1[i])*100)/attributes1[i])
            difference2.append(stats.ks_2samp(neg_x[:, i], pos_x[:, i]))
            print(i)

        print(attributes1)
        print(attributes2)
        print(difference)
        print(difference2)
        for i in range(len(difference2)):
            if difference2[i][0] > 0.1 and difference2[i][1] < 0.001:
                print(i, difference2[i])



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
    with open('part3_pricing_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    return trained_model


if __name__ == "__main__":
    #test = load_model()
    #x, y, claims_raw, y1 = test.load_data("part3_training_data.csv")
    #print(test.predict_claim_probability(x).shape)
    #test.base_classifier.evaluate_architecture(True)

    test = PricingModel()
    x, y, claims_raw, y1 = test.load_data("part3_training_data.csv")
    print(x.shape)
    x.drop(columns=["drv_sex2"], inplace=True)
    x.dropna(how="any", inplace=True)
    x = test.integer_encode(x)
    min_max_scaler = preprocessing.MinMaxScaler()
    x = min_max_scaler.fit_transform(x)

    test.evaluate_input1(x)

    """
    test = PricingModel()
    x, y, claims_raw, y1 = test.load_data("part3_training_data.csv")


    x = test._preprocessor(x)
    print(x.shape, y.shape, claims_raw.shape, y1.shape)
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
    #test.base_classifier.evaluate_architecture(True)

    """
    """
    list = [2,15,17,18,21,22,23,25,26]
    print([x.columns[i] for i in list])
    #x = test.integer_encode(x)
    x = test._preprocessor(x)
    print(x[:,2])
    #x = test._preprocessor(x)
    y = y.to_numpy()
    print(x.shape)
    print(y.shape)
    #split = test.separate_pos_neg(x, y)

    #file = open('Part3_Split_norm.pickle', 'wb')
    #pickle.dump(split, file)
    #file.close()

    file = open('Part3_Split_norm.pickle', 'rb')
    split = pickle.load(file)
    file.close()

    print(split[0][0].shape, split[1][0].shape)
    test.evaluate_input3(x, y, split)
    """

"""
    for i in range(x.shape[0]):
        try:
            z = np.array(x[i,:])
            z.astype(np.float)
            #x_new = np.append(x_new, [z], axis=0)
        except ValueError:
            print(x[i,:])
            print(i)
            count+=1
    print(count)
    
    #print(x.shape[0] - x_new.shape[0])

"""