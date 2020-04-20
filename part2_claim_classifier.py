import numpy as np
import pickle
import matplotlib.pyplot as plt
import copy
import pandas as pd


import random
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn import metrics
#from pytorchtools import EarlyStopping
from sklearn import preprocessing
from scipy import stats
import math
from matplotlib.ticker import MaxNLocator

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss

def analyse(model, data_x, data_y):
    # data_x and data_y are numpy array-of-arrays matrices
    X = torch.Tensor(data_x)
    Y = torch.ByteTensor(data_y)   # a Tensor of 0s and 1s
    oupt = model(X)            # a Tensor of floats
    pred_y = oupt >= 0.5       # a Tensor of 0s and 1s
    num_correct = torch.sum(Y==pred_y)  # a Tensor
    acc = (num_correct.item() * 100.0 / len(data_y))  # scalar
    return (acc, pred_y, oupt)

"""
class Insurance_NN(nn.Module):
    def __init__(self):
        super(Insurance_NN, self).__init__()

        self.apply_layers = nn.Sequential(
            # 2 fully connected hidden layers of 8 neurons goes to 1
            # 9 - (100 - 10) - 1
            nn.Linear(9, 100),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(100, 10),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(10, 1),
            nn.Sigmoid()
        )

    # Defining the forward pass
    def forward(self, x):
        x = self.apply_layers(x)
        return x.view(len(x))
"""

def weighted_binary_cross_entropy(sigmoid_x, targets, pos_weight, weight=None, size_average=True, reduce=True):
    """
    Args:
        sigmoid_x: predicted probability of size [N,C], N sample and C Class. Eg. Must be in range of [0,1], i.e. Output from Sigmoid.
        targets: true value, one-hot-like vector of size [N,C]
        pos_weight: Weight for postive sample
    """
    if not (targets.size() == sigmoid_x.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(targets.size(), sigmoid_x.size()))

    loss = -pos_weight* targets * sigmoid_x.log() - (1-targets)*(1-sigmoid_x).log()

    if weight is not None:
        loss = loss * weight

    if not reduce:
        return loss
    elif size_average:
        return loss.mean()
    else:
        return loss.sum()

class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight=1, weight=None, PosWeightIsDynamic= False, WeightIsDynamic= False, size_average=True, reduce=True):
        """
        Args:
            pos_weight = Weight for postive samples. Size [1,C]
            weight = Weight for Each class. Size [1,C]
            PosWeightIsDynamic: If True, the pos_weight is computed on each batch. If pos_weight is None, then it remains None.
            WeightIsDynamic: If True, the weight is computed on each batch. If weight is None, then it remains None.
        """
        super().__init__()

        self.register_buffer('weight', weight)
        self.register_buffer('pos_weight', pos_weight)
        self.size_average = size_average
        self.reduce = reduce
        self.PosWeightIsDynamic = PosWeightIsDynamic

    def forward(self, input, target):
        # pos_weight = Variable(self.pos_weight) if not isinstance(self.pos_weight, Variable) else self.pos_weight
        if self.PosWeightIsDynamic:
            positive_counts = target.sum(dim=0)
            nBatch = len(target)
            self.pos_weight = (nBatch - positive_counts)/(positive_counts +1e-5)

        if self.weight is not None:
            # weight = Variable(self.weight) if not isinstance(self.weight, Variable) else self.weight
            return weighted_binary_cross_entropy(input, target,
                                                 self.pos_weight,
                                                 weight=self.weight,
                                                 size_average=self.size_average,
                                                 reduce=self.reduce)
        else:
            return weighted_binary_cross_entropy(input, target,
                                                 self.pos_weight,
                                                 weight=None,
                                                 size_average=self.size_average,
                                                 reduce=self.reduce)



    def __iter__(self):
        return self

    def __next__(self):
        if self.ptr + self.batch_size > self.num_items:
            self.rnd.shuffle(self.indices)
            self.ptr = 0
            raise StopIteration  # exit calling for-loop
        else:
            result = self.indices[self.ptr:self.ptr + self.batch_size]
            self.ptr += self.batch_size
            return result


class ClaimClassifier():

    def __init__(self, model):
        """
        Feel free to alter this as you wish, adding instance variables as
        necessary. 
        """
        self.fitted_model = model;

        self.train_data = 0;
        self.test_data = 0;
        self.val_data = 0;

        self.classes_ = np.array([0,1])

    def load_data(self, filename, drop_extra = False):
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
        # load data to single 2D array
        dat = pd.read_csv("part2_training_data.csv")
        if drop_extra:
            dat = dat.drop(columns=["drv_age1", "vh_cyl", "pol_bonus"])
            #dat = dat.drop(columns=["vh_sale_begin", "vh_sale_end", "vh_age"])
        x = dat.drop(columns=["claim_amount", "made_claim"])
        y = dat["made_claim"]
        y2 = dat["claim_amount"]

        return x, y, y2

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
        ndarray
            A clean data set that is used for training and prediction.
        """
        # YOUR CODE HERE

        if not isinstance(X_raw, np.ndarray):
            X_raw = X_raw.to_numpy(dtype=np.float)

        min_max_scaler = preprocessing.MinMaxScaler()
        X_raw = min_max_scaler.fit_transform(X_raw)

        return X_raw.astype(np.float32)

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


        fig, ax1 = plt.subplots(figsize=(11, 4))

        # type of plot
        ax1.boxplot(attributes)
        labels = ['drv_age1', 'vh_age', 'vh_cyl', 'vh_din', 'pol_bonus', 'vh_sl_b',
                  'vh_sl_e', 'vh_value', 'vh_speed']

        self.set_axis_style(ax1, labels)

        plt.subplots_adjust(bottom=0.15, wspace=0.05)
        # plt.show()
        plt.xlabel("Attribute Type")
        plt.ylabel("Attribute Value")

        plt.savefig("box.pdf", bbox_inches='tight')

        ####################

        plt.cla()
        ax1.violinplot(attributes)

        labels = ['drv_age1', 'vh_age', 'vh_cyl', 'vh_din', 'pol_bonus',
                  'vh_sl_b', 'vh_sl_e', 'vh_value', 'vh_speed']

        self.set_axis_style(ax1, labels)

        plt.subplots_adjust(bottom=0.15, wspace=0.05)
        plt.xlabel("Attribute Type")
        plt.ylabel("Attribute Value")


        plt.savefig("violin.pdf", bbox_inches='tight')

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

        fig, axs = plt.subplots(2, figsize=(11, 8))

        # type of plot
        axs[0].boxplot(attributes1, showfliers=False)
        axs[1].boxplot(attributes2, showfliers=False)
        labels = ['drv_age1', 'vh_age', 'vh_cyl', 'vh_din', 'pol_bonus', 'vh_sl_b',
                  'vh_sl_e', 'vh_value', 'vh_speed']

        self.set_axis_style(axs[0], labels)
        self.set_axis_style(axs[1], labels)

        # plt.show()
        axs[0].set(xlabel="", ylabel="Attribute Value")
        axs[0].set_title("No Claim")
        axs[1].set(xlabel="Attribute Type", ylabel="Attribute Value")
        axs[1].set_title("Claim")

        #plt.subplots_adjust(bottom=0.15, wspace=0.05)
        plt.savefig("compare_box.pdf", bbox_inches='tight')

    def evaluate_input3(self, x, y):
        """
        Function to evaluate data loaded from file

        """

        # Separate positive and negative results

        (neg_x, neg_y), (pos_x, pos_y) = self.separate_pos_neg(x, y)
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
            if difference2[i][0] > 0.1:
                print(i, difference2[i])

    def initialTrain(self, model, train_x, train_y, val_x, val_y, with_weight = False):
        """
        Initial training algorithm used. The with_weight argument is boolean
        and defines whether or not the We
        """
        if with_weight == True:
            pos_weight = torch.Tensor([(900 / 100)])
            criterion = WeightedBCELoss(pos_weight)
        else:
            criterion = nn.BCELoss()

        optimiser = torch.optim.SGD(model.parameters(), lr=0.01)

        batch_size = 32
        num_epochs = 250

        # to track the training loss as the model trains
        train_losses = []
        # to track the validation loss as the model trains
        valid_losses = []
        # to track the average training loss per epoch as the model trains
        avg_train_losses = []
        # to track the average validation loss per epoch as the model trains
        avg_valid_losses = []

        epochs = []

        for epoch in range(num_epochs):
            model.train()
            shuffled_train_x, shuffled_train_y = shuffle(train_x, train_y,
                                                         random_state=0)
            shuff_val_x, shuff_val_y = shuffle(val_x, val_y, random_state=0)

            x_batches = torch.split(shuffled_train_x, batch_size, dim=0)
            y_batches = torch.split(shuffled_train_y, batch_size, dim=0)
            x_val_batches = torch.split(shuff_val_x, batch_size, dim=0)
            y_val_batches = torch.split(shuff_val_y, batch_size, dim=0)

            # TRAIN MODEL
            for batch_i in range(len(x_batches)):

                batch_data = x_batches[batch_i]
                batch_label = y_batches[batch_i]

                # Forward pass: compute predicted outputs
                optimiser.zero_grad()
                batch_output = model(batch_data)

                # Backward Pass: Calculate loss by comparing ground truth to predictions on batch
                batch_loss = criterion(batch_output, batch_label)
                batch_loss.backward()
                optimiser.step()
                train_losses.append(batch_loss.item())

            # VALIDATE MODEL
            model.eval() #prep for evaluation

            for val_i in range(len(x_val_batches)):
                # Predict on validiation batch
                output = model(x_val_batches[val_i])
                # Calculate loss and save in list
                loss = criterion(output, y_val_batches[val_i])
                valid_losses.append(loss.item())

            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)
            epochs.append(epoch+1)

            train_losses = []
            valid_losses = []

            print("Epoch = %d, Loss = %f" % (epoch + 1, batch_loss.item()))
            acc = analyse(model, val_x, val_y.numpy())[0]
            print("Validation Accuracy = ", acc)

        fig, ax1 = plt.subplots()
        ax1.plot(epochs, avg_train_losses, 'b', label="Training Loss")
        ax1.plot(epochs, avg_valid_losses, 'r', label="Validation Loss")
        #ax1.plot(epochs, epochs, 'g')
        legend = ax1.legend()
        legend.get_frame().set_edgecolor('k')
        plt.xlabel("Epoch")
        plt.ylabel("Average Loss")
        plt.xlim(0, num_epochs)
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.savefig("initial_loss_plot.pdf", bbox_inches='tight')

        return model

    def weightedTrain(self, model, train_x, train_y, val_x, val_y, with_weight = True):
        # Weighted version of train
        # https://discuss.pytorch.org/t/unclear-about-weighted-bce-loss/21486
        # https://github.com/pytorch/pytorch/issues/5660
        if with_weight:
            pos_weight = torch.Tensor([(900 / 100)])
            criterion = WeightedBCELoss(pos_weight)
        else:
            criterion = nn.BCELoss()

        batch_size = 5
        num_epochs = 50

        #print(torch.sum(train_y)/train_y.shape[0])
        optimiser = torch.optim.SGD(model.parameters(), lr=0.1)
        #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimiser, max_lr=0.1, steps_per_epoch=math.ceil((len(train_x)/batch_size)),epochs=num_epochs)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, patience=10)

        # to track the training loss as the model trains
        train_losses = []
        # to track the validation loss as the model trains
        valid_losses = []
        # to track the average training loss per epoch as the model trains
        avg_train_losses = []
        # to track the average validation loss per epoch as the model trains
        avg_valid_losses = []
        epochs = []
        early_stopping = EarlyStopping(patience=100, verbose=True)



        for epoch in range(num_epochs):
            model.train()
            shuffled_train_x, shuffled_train_y = shuffle(train_x, train_y,
                                                         random_state=0)
            shuff_val_x, shuff_val_y = shuffle(val_x, val_y, random_state=0)

            x_batches = torch.split(shuffled_train_x, batch_size, dim=0)
            y_batches = torch.split(shuffled_train_y, batch_size, dim=0)
            x_val_batches = torch.split(shuff_val_x, batch_size, dim=0)
            y_val_batches = torch.split(shuff_val_y, batch_size, dim=0)

            for param_group in optimiser.param_groups:
                print("\nLearning Rate = ",param_group['lr'])

            # TRAIN MODEL
            for batch_i in range(len(x_batches)):

                batch_data = x_batches[batch_i]
                batch_label = y_batches[batch_i]

                # Forward pass: compute predicted outputs
                optimiser.zero_grad()
                batch_output = model(batch_data)

                # Backward Pass: Calculate loss by comparing ground truth to predictions on batch
                batch_loss = criterion(batch_output, batch_label)
                batch_loss.backward()
                optimiser.step()
                # Signal OneCycleLR adaptive LR
                #scheduler.step()
                train_losses.append(batch_loss.item())


            # VALIDATE MODEL
            model.eval() #prep for evaluation

            for val_i in range(len(x_val_batches)):
                # Predict on validiation batch
                output = model(x_val_batches[val_i])
                # Calculate loss and save in list
                loss = criterion(output, y_val_batches[val_i])
                valid_losses.append(loss.item())

            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)
            epochs.append(epoch+1)
            # Signal ReduceLROnPlateau adaptive LR with validation loss
            scheduler.step(valid_loss)

            train_losses = []
            valid_losses = []

            print("Epoch = %d, Loss = %f" % (epoch + 1, batch_loss.item()))
            acc = analyse(model, val_x, val_y.numpy())[0]
            print("Validation Accuracy = ", acc)

            # early_stopping needs the validation loss to check if it has decresed,
            # and if it has, it will make a checkpoint of the current model
            early_stopping(valid_loss, model)

            if early_stopping.early_stop:
                print("Early Stopping")
                break

            model.load_state_dict(torch.load('checkpoint.pt'))

        fig, ax1 = plt.subplots()
        ax1.plot(epochs, avg_train_losses, 'b', label="Training Loss")
        ax1.plot(epochs, avg_valid_losses, 'r', label="Validation Loss")
        # ax1.plot(epochs, epochs, 'g')
        legend = ax1.legend()
        legend.get_frame().set_edgecolor('k')
        plt.xlabel("Epoch")
        plt.ylabel("Average Loss")
        plt.xlim(0, num_epochs)
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.savefig("initial_loss_plot.pdf", bbox_inches='tight')

        return model



    def downsampleTrain(self, model, train_x, train_y, val_x, val_y):

        criterion = nn.BCELoss()

        num_epochs = 100
        batch_size = 10

        optimiser = torch.optim.SGD(model.parameters(), lr=0.1)
        #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimiser, max_lr=0.1,steps_per_epoch=math.ceil((len(train_x)/batch_size)),epochs=num_epochs)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, patience=10)

        # to track the training loss as the model trains
        train_losses = []
        # to track the validation loss as the model trains
        valid_losses = []
        # to track the average training loss per epoch as the model trains
        avg_train_losses = []
        # to track the average validation loss per epoch as the model trains
        avg_valid_losses = []
        epochs = []

        early_stopping = EarlyStopping(patience=100, verbose=True)

        # Separate into positive and negative samples
        (neg_train_x, neg_train_y), (pos_train_x, pos_train_y) = \
            self.separate_pos_neg(train_x, train_y)

        print(len(pos_train_y))
        print(pos_train_x.shape)

        print(len(neg_train_y))

        neg_train_x, neg_train_y = shuffle(neg_train_x, neg_train_y)


        for epoch in range(num_epochs):
            model.train()

            neg_train_x, neg_train_y = shuffle(neg_train_x, neg_train_y)

            # 2004, 2508, 3012
            train_x_new = np.concatenate((neg_train_x[:int(1.0*len(pos_train_x))], pos_train_x))
            train_y_new = np.concatenate((neg_train_y[:int(1.0*len(pos_train_x))], pos_train_y))

            # concat first 1668 of this matrix to pos vals then proceed as if theyr're train_x and train_y
            shuffled_train_x, shuffled_train_y = shuffle(train_x_new, train_y_new,
                                                         random_state=0)
            shuff_val_x, shuff_val_y = shuffle(val_x, val_y, random_state=0)

            shuffled_train_x = torch.from_numpy(shuffled_train_x)
            shuffled_train_y = torch.from_numpy(shuffled_train_y)

            x_batches = torch.split(shuffled_train_x, batch_size, dim=0)
            y_batches = torch.split(shuffled_train_y, batch_size, dim=0)
            x_val_batches = torch.split(shuff_val_x, batch_size, dim=0)
            y_val_batches = torch.split(shuff_val_y, batch_size, dim=0)

            for param_group in optimiser.param_groups:
                print("\nLearning Rate = ", param_group['lr'])

            for batch_i in range(len(x_batches)):

                batch_data = x_batches[batch_i]
                batch_label = y_batches[batch_i]

                # Perform gradient decent algorithm to reduce loss
                optimiser.zero_grad()
                batch_output = model(batch_data)

                # Calculate loss by comparing ground truth to predictions on batch
                batch_loss = criterion(batch_output, batch_label)
                batch_loss.backward()
                optimiser.step()
                # Signal OneCycleLR adaptive LR
                #scheduler.step()
                train_losses.append(batch_loss.item())

            # VALIDATE MODEL
            model.eval()  # prep for evaluation

            for val_i in range(len(x_val_batches)):
                # Predict on validiation batch
                output = model(x_val_batches[val_i])
                # Calculate loss and save in list
                loss = criterion(output, y_val_batches[val_i])
                valid_losses.append(loss.item())

            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)
            epochs.append(epoch+1)
            # Signal ReduceLROnPlateau adapaptive LR with validation loss
            scheduler.step(valid_loss)

            train_losses = []
            valid_losses = []

            print("Epoch = %d, Loss = %f" % (epoch + 1, batch_loss.item()))
            acc = analyse(model, val_x, val_y.numpy())[0]
            print("Validation Accuracy = ", acc)

            # early_stopping needs the validation loss to check if it has decresed,
            # and if it has, it will make a checkpoint of the current model
            early_stopping(valid_loss, model)

            if early_stopping.early_stop:
                print("Early Stopping")
                break

            model.load_state_dict(torch.load('checkpoint.pt'))

        fig, ax1 = plt.subplots()
        ax1.plot(epochs, avg_train_losses, 'b', label="Training Loss")
        ax1.plot(epochs, avg_valid_losses, 'r', label="Validation Loss")
        # ax1.plot(epochs, epochs, 'g')
        legend = ax1.legend()
        legend.get_frame().set_edgecolor('k')
        plt.xlabel("Epoch")
        plt.ylabel("Average Loss")
        plt.xlim(0, num_epochs)
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.savefig("initial_loss_plot.pdf", bbox_inches='tight')

        return model


    def weightedUpsampleTrain(self, model, train_x, train_y, val_x, val_y):

        criterion = nn.BCELoss()
        #print(train_x,train_y)

        num_epochs = 100
        batch_size = 16

        optimiser = torch.optim.SGD(model.parameters(), lr=0.1)
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimiser, max_lr=0.1,steps_per_epoch=math.ceil((len(train_x)/batch_size)),epochs=num_epochs)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, patience=10)

        # to track the training loss as the model trains
        train_losses = []
        # to track the validation loss as the model trains
        valid_losses = []
        # to track the average training loss per epoch as the model trains
        avg_train_losses = []
        # to track the average validation loss per epoch as the model trains
        avg_valid_losses = []

        epochs = []

        early_stopping = EarlyStopping(patience=250, verbose=True)

        for epoch in range(num_epochs):
            model.train()

            shuffled_train_x, shuffled_train_y = shuffle(train_x, train_y,
                                                         random_state=0)
            shuff_val_x, shuff_val_y = shuffle(val_x, val_y, random_state=0)

            #shuffled_train_x = torch.from_numpy(shuffled_train_x)
            #shuffled_train_y = torch.from_numpy(shuffled_train_y)

            x_batches = torch.split(shuffled_train_x, batch_size, dim=0)
            y_batches = torch.split(shuffled_train_y, batch_size, dim=0)
            x_val_batches = torch.split(shuff_val_x, batch_size, dim=0)
            y_val_batches = torch.split(shuff_val_y, batch_size, dim=0)

            for param_group in optimiser.param_groups:
                print("\nLearning Rate = ", param_group['lr'])

            for batch_i in range(len(x_batches)):
                batch_data = x_batches[batch_i]
                batch_label = y_batches[batch_i]

                # Perform gradient decent algorithm to reduce loss
                optimiser.zero_grad()
                batch_output = model(batch_data)

                # Calculate loss by comparing ground truth to predictions on batch
                batch_loss = criterion(batch_output, batch_label)
                batch_loss.backward()
                optimiser.step()
                # Signal OneCycleLR adaptive LR
                # scheduler.step()
                train_losses.append(batch_loss.item())

            # VALIDATE MODEL
            model.eval()  # prep for evaluation

            for val_i in range(len(x_val_batches)):
                # Predict on validiation batch
                output = model(x_val_batches[val_i])
                # Calculate loss and save in list
                loss = criterion(output, y_val_batches[val_i])
                valid_losses.append(loss.item())

            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)
            epochs.append(epoch+1)

            # Signal ReduceLROnPlateau adapaptive LR with validation loss
            scheduler.step(valid_loss)

            train_losses = []
            valid_losses = []

            print("Epoch = %d, Loss = %f" % (epoch + 1, batch_loss.item()))
            acc = analyse(model, val_x, val_y.numpy())[0]
            print("Validation Accuracy = ", acc)

            # early_stopping needs the validation loss to check if it has decresed,
            # and if it has, it will make a checkpoint of the current model
            early_stopping(valid_loss, model)

            if early_stopping.early_stop:
                print("Early Stopping")
                break

            model.load_state_dict(torch.load('checkpoint.pt'))

        fig, ax1 = plt.subplots()
        ax1.plot(epochs, avg_train_losses, 'b', label="Training Loss")
        ax1.plot(epochs, avg_valid_losses, 'r', label="Validation Loss")
        # ax1.plot(epochs, epochs, 'g')
        legend = ax1.legend()
        legend.get_frame().set_edgecolor('k')
        plt.xlabel("Epoch")
        plt.ylabel("Average Loss")
        plt.xlim(0, num_epochs)
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.savefig("initial_loss_plot.pdf", bbox_inches='tight')

        return model

    def separate_data(self, X_raw, Y_raw, y2 = 0):
        """
        Separate data into training and test data in 85:15 ratio. The training
        data is then further partitioned to make validation set in fit( )
        class method, resulting in 70:15:15 split of train:validation:test
        data.

        3rd optional argument should be inputed when using weightedUpsampleTrain()
        to train network
        """
        # Condition where weighted upsampling is being used (so claim_amount data
        # is needed)
        if not isinstance(y2, int):
            Y_raw = pd.concat([Y_raw, y2], axis=1)

        train_x, test_x, train_y, test_y = train_test_split(X_raw, Y_raw,
                                                              test_size=0.15)
        if not isinstance(y2, int):
            test_y = test_y.drop(columns= "claim_amount")

        # Save split for evaluation later
        if not isinstance(test_y, np.ndarray):
            test_y = test_y.to_numpy(dtype=np.float)

        self.test_data = (self._preprocessor(test_x), test_y)

        return (train_x, train_y), (test_x, test_y)


    def upsample_generate(self, train_x, train_y, claim_amount):
        """
        Upsample training data for UpsampleTrain weighted
        """

        # Add claim_amount to the train_x as new column
        train_x = np.column_stack((train_x, claim_amount))
        # Separate into pos and negative
        (neg_x, neg_y), (pos_x, pos_y) = self.separate_pos_neg(train_x,
                                                               train_y)
        #print(neg_x.shape, neg_y.shape)
        #print(pos_x.shape, pos_y.shape)

        # Remove additional claim column for neg_x
        neg_x = neg_x[:, :-1]

        # Remove and save additional claim column for pos_x
        claim_amount = pos_x[:, -1:]
        pos_x = pos_x[:, :-1]
        #print(pos_x)
        # Normalise claim amount such that their sum is 1
        claim_multiplier = claim_amount / sum(claim_amount)

        # Work out how many times to duplicate each row
        # (sum of claim multiplier should equal len(neg_y))
        claim_multiplier = np.ceil(claim_multiplier * len(neg_y))
        claim_multiplier = claim_multiplier.astype(int)
        #print(sum(claim_multiplier), len(neg_y))
        #print(claim_multiplier.shape)

        # Do duplication
        new_pos_x = np.empty((0, pos_x.shape[1]), np.float32)
        for i, row in enumerate(pos_x):
            #print(row)
            for i2 in range(claim_multiplier[i][0]):
                new_pos_x = np.vstack((new_pos_x, row))
                #print(row)

        #print(new_pos_x)
        #print(neg_x.shape, new_pos_x.shape)
        new_pos_y = np.ones(len(new_pos_x))

        train_x = np.append(neg_x, new_pos_x, axis=0)
        train_y = np.append(neg_y, new_pos_y, axis=0)
        #print(train_x.shape)
        train_x = train_x.astype(np.float32)
        train_y = train_y.astype(np.float32)
        return train_x, train_y


    def fit(self, X_raw, y_raw, save_model = False):
        """Classifier training function.

        Here you will implement the training function for your classifier.

        Parameters
        ----------
        X_raw : ndarray
            An array, this is the raw data as downloaded
        y_raw : ndarray (optional)
            A one dimensional array, this is the binary target variable

        Returns
        -------
        self: (optional)
            an instance of the fitted model

         ======********* PLEASE NOTE **********=======

         When calling this function for training with weightedUpsampleTrain
         the y variable will contain two columns - one for the label and one
         for the claim amount. This concatenation is done when separate_data has
         the extra optional 3rd argument of claim_amount loaded.

        """

        # REMEMBER TO HAVE THE FOLLOWING LINE SOMEWHERE IN THE CODE
        # X_clean = self._preprocessor(X_raw)
        # YOUR CODE HERE
        #X_raw = X_raw.to_numpy()
        if not isinstance(y_raw, np.ndarray):
            y_raw = y_raw.to_numpy(dtype=np.float32)

        if not isinstance(X_raw, np.ndarray):
            X_raw = X_raw.to_numpy(dtype=np.float)

        X_clean = self._preprocessor(X_raw)

        # Split data into training and val
        # making val 17.647% of original makes it a total of 15% of original
        # data --> see separate_data( ) class method
        train_x, val_x, train_y, val_y = train_test_split(X_clean, y_raw,
                                                            test_size = 0.17647)

        # FOR WEIGHTED UPSAMPLE TRAIN
        if (train_y.ndim > 1):
            claim_amount = train_y[:,1]
            train_y = train_y[:, 0]
            val_y = val_y[:,0]
            # generate upsampled training data.
            train_x, train_y = self.upsample_generate(train_x, train_y, claim_amount)

        # Save split for later evaluation
        self.train_data = (train_x, train_y)
        self.val_data = (val_x, val_y)

        print((train_x.shape, train_y.shape), (val_x.shape, val_y.shape))

        train_x = torch.from_numpy(train_x)
        train_y = torch.from_numpy(train_y)
        val_x = torch.from_numpy(val_x)
        val_y = torch.from_numpy(val_y)

        print(self.fitted_model)
        print(self.fitted_model(train_x).shape)

        self.fitted_model.train()

        # ----------- MAKE METHOD SELECTION -------------
        #self.fitted_model = self.initialTrain(self.fitted_model, train_x, train_y, val_x, val_y, True)
        #self.fitted_model = self.weightedTrain(self.fitted_model, train_x, train_y, val_x, val_y)
        #self.fitted_model = self.downsampleTrain(self.fitted_model, train_x, train_y, val_x, val_y)
        self.fitted_model = self.weightedUpsampleTrain(self.fitted_model, train_x, train_y, val_x, val_y)

        if save_model:
            self.save_model()

        return self



    def predict(self, X_raw):
        """Classifier probability prediction function.

        Here you will implement the predict function for your classifier.

        Parameters
        ----------
        X_raw : ndarray
            An array, this is the raw data as downloaded

        Returns
        -------
        ndarray
            A one dimensional array of the same length as the input with
            values corresponding to the probability of beloning to the
            POSITIVE class (that had accidents)
        """

        # REMEMBER TO HAVE THE FOLLOWING LINE SOMEWHERE IN THE CODE
        # X_clean = self._preprocessor(X_raw)

        # YOUR CODE HERE

        #X_raw = X_raw.to_numpy()

        if not isinstance(X_raw, np.ndarray):
            X_raw= X_raw.to_numpy(dtype=np.float)

        X_clean = self._preprocessor(X_raw)
        #self.fitted_model = load_model().fitted_model

        X_test = torch.Tensor(X_clean)
        oupt = self.fitted_model(X_test)  # a Tensor of floats
        pred_y = oupt >= 0.5  # a Tensor of 0s and 1s

        return pred_y.numpy()


    def predict_proba(self, X_raw):
        """
        Used in Part 3
        """

        if not isinstance(X_raw, np.ndarray):
            X_raw= X_raw.to_numpy(dtype=np.float)

        X_clean = self._preprocessor(X_raw)
        #self.fitted_model = load_model().fitted_model

        X_test = torch.Tensor(X_clean)
        oupt = self.fitted_model(X_test)  # a Tensor of floats
        pos = oupt.detach().numpy()
        neg = np.ones(len(pos))
        neg = neg - pos

        # (N,2) array with first column being prob of 0 and second being
        # prob of 1
        ans = np.column_stack((neg, pos))

        return ans


    def evaluate_architecture(self, with_test = False):
        """Architecture evaluation utility.

        Populate this function with evaluation utilities for your
        neural network.

        You can use external libraries such as scikit-learn for this
        if necessary.
        """
        train_x, train_y = self.train_data
        print("Training Data Shape = ", train_x.shape, train_y.shape)
        val_x, val_y = self.val_data
        print("Validation Data Shape = ", val_x.shape, val_y.shape)

        # Calculate and print accuracies based on model predictions
        acc1 = analyse(self.fitted_model, train_x, train_y)
        acc2 = analyse(self.fitted_model, val_x, val_y)
        #print(train_x, train_y)
        print("Train Accuracy = ", acc1[0])
        auc_score1 = metrics.roc_auc_score(train_y, acc1[2].detach().numpy())
        print("Train AUC Score = ", auc_score1)
        print("Validation Accuracy = ", acc2[0])
        auc_score2 = metrics.roc_auc_score(val_y, acc2[2].detach().numpy())
        print("Validation AUC Score = ", auc_score2)

        labels = ['No Claim', 'Claim']

        if with_test:
            test_x, test_y = self.test_data
            print("Test Data Shape = ", test_x.shape, test_y.shape)
            acc3 = analyse(self.fitted_model, test_x, test_y.reshape((len(test_y),)))
            print("Test Accuracy = ", acc3[0])
            auc_score3 = metrics.roc_auc_score(test_y,
                                              acc3[2].detach().numpy())
            print("Test AUC Score = ", auc_score3)

            f, (ax1, ax2, ax3) = plt.subplots(1, 3)
            confusion_test = metrics.confusion_matrix(test_y, acc3[1].numpy(),
                                                  normalize='true')
            # Plot confusion for test data
            metrics.ConfusionMatrixDisplay(confusion_test, labels).plot(ax=ax3)
            ax3.set_title("Test Set", fontsize=17)
            ax3.set_ylabel("")
            plot_width = 15
        else:
            f, (ax1, ax2) = plt.subplots(1, 2)
            plot_width = 10

        # Construct training and validation normalised confusion matricies
        confusion_train = metrics.confusion_matrix(train_y, acc1[1].numpy(),
                                              normalize='true')
        confusion_val = metrics.confusion_matrix(val_y, acc2[1].numpy(),
                                              normalize='true')

        # Plot training and validation set confusion matricies
        metrics.ConfusionMatrixDisplay(confusion_train, labels).plot(ax=ax1)
        ax1.set_title("Training Set", fontsize=17)
        metrics.ConfusionMatrixDisplay(confusion_val, labels).plot(ax=ax2)
        ax2.set_title("Validation Set", fontsize=17)
        ax2.set_ylabel("")

        plt.gcf().set_size_inches(plot_width+1, 5)
        plt.savefig("confusion_matrix.pdf", bbox_inches='tight')
        plt.show()

        return

    def save_model(self):
        # Please alter this file appropriately to work in tandem with your load_model function below
        with open('part2_claim_classifier.pickle', 'wb') as target:
            pickle.dump(self, target)



def load_model():
    # Please alter this section so that it works in tandem with the save_model method of your class
    with open('part2_claim_classifier.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    return trained_model

# ENSURE TO ADD IN WHATEVER INPUTS YOU DEEM NECESSARRY TO THIS FUNCTION
def ClaimClassifierHyperParameterSearch():
    """Performs a hyper-parameter for fine-tuning the classifier.

    Implement a function that performs a hyper-parameter search for your
    architecture as implemented in the ClaimClassifier class. 

    The function should return your optimised hyper-parameters. 
    """

    return  # Return the chosen hyper parameters


class Insurance_NN(nn.Module):
    def __init__(self):
        super(Insurance_NN, self).__init__()

        self.apply_layers = nn.Sequential(
            # 2 fully connected hidden layers of 8 neurons goes to 1
            # 9/6 - (100 - 10) - 1
            nn.Linear(9, 30),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(),
            #nn.Linear(20, 5),
            #nn.LeakyReLU(inplace=True),
            #nn.Dropout(),
            nn.Linear(30, 1),
            nn.Sigmoid()
        )

    # Defining the forward pass
    def forward(self, x):
        x = self.apply_layers(x)
        return x.view(len(x))

if __name__ == "__main__":
    #test = load_model()
    #test.evaluate_architecture(True)

    test = ClaimClassifier(Insurance_NN())
    x, y, y2 = test.load_data("part2_training_data.csv")

    #x = test._preprocessor(x)
    #test.evaluate_input1(x)

    train_data, test_data = test.separate_data(x, y, y2)
    test.fit(train_data[0], train_data[1], True)
    #print(train_data[0].shape, train_data[1].shape)
    #print(test_data[0].shape, test_data[1].shape)

    #train_x, train_y = (train_data[0].to_numpy(), train_data[1].to_numpy())
    #print(type(train_x), type(train_y))
    #claim_amount = train_y[:, 1]
    #train_y = train_y[:, 0]

    #ans1, ans2 = test.upsample_generate(train_x, train_y, claim_amount)
    #print(ans1, ans2)

    #list = [1, 3, 5, 6, 7, 8]
    #print([x.columns[i] for i in list])
    #x = x.to_numpy(dtype=float)
    #test.evaluate_input3(x, y.to_numpy(dtype=float))


    #train_data, test_data = test.separate_data(x, y)
    #test.fit(train_data[0], train_data[1], True)
    predictions_test = test.predict(pd.DataFrame(test_data[0]))


    confusion_test = metrics.confusion_matrix(test.test_data[1], predictions_test,
                                             normalize='true')

    labels = ['No Claim', 'Claim']
    metrics.ConfusionMatrixDisplay(confusion_test, labels).plot()

    test.evaluate_architecture(True)
    #test.evaluate_architecture()

    """
    #test.evaluate_input3(x, y)
    x_clean = test._preprocessor(x)
    #print(x_clean.shape)
    #test.fit(x, y)
    test.evaluate_input3(x, y)
    
    data_set = np.genfromtxt("part2_training_data.csv", dtype=float, delimiter=',', skip_header=1)
    num_att = len(data_set[0])  # number of parameters

    claims = np.array(data_set[:, (num_att - 1)], dtype=np.float32)
    claim_amount = np.array(data_set[:, (num_att - 2)], dtype=np.float32)
    print(max(claim_amount))

    amounts_list = []

    for i in range(len(claim_amount)):
        if claims[i] == 1:
            amounts_list.append(claim_amount[i])

    print(amounts_list)

    fig, ax1 = plt.subplots(figsize=(4, 4), sharey=True)

    # type of plot
    ax1.boxplot(amounts_list)

    labels = ['Claim Amount']
    test = ClaimClassifier()
    test.set_axis_style(ax1, labels)

    plt.subplots_adjust(bottom=0.15, wspace=0.05)
    # plt.show()
    plt.xlabel("")
    plt.ylabel("Amount")

    plt.show()
    """