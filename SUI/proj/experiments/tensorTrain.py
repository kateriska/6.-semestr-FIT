import numpy as np
import copy
import matplotlib.pyplot as plt
import scipy.stats
import torch

import torch
import torch.nn.functional as F

def batch_provider(xs, targets, batch_size=10):
    data_torch = torch.from_numpy(xs).float()
    targets_torch = torch.from_numpy(targets).float()

    dataset = torch.utils.data.TensorDataset(data_torch, targets_torch)
    # create minibatches
    dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
    return dataloader

class FullLogisticRegression(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.parameter.Parameter(torch.tensor([1.0]))
        self.b = torch.nn.parameter.Parameter(torch.tensor([0.0]))
        self.hidden = torch.nn.Linear(10, 1)

    def forward(self, x):
        x_numpy_array = (self.hidden(x)).detach().numpy()
        x_numpy_array_converted = x_numpy_array.flatten()
        x = torch.from_numpy(x_numpy_array_converted).float()
        x = torch.sigmoid(self.w*x + self.b)
        return x

    def prob_class_1(self, x):
        prob = self(torch.from_numpy(x))
        return prob.detach().numpy()

def evaluateFullLogisticRegression(classifier, inputs, targets ):
    accuracy = train_all_fea_llr(100, 0.01, 2, inputs, targets)[2]
    return max(accuracy)

def train_all_fea_llr(nb_epochs, lr, batch_size, inputs, targets):

    model = FullLogisticRegression()
    best_model = copy.deepcopy(model)
    losses = []
    accuracies = []
    epochs_list = []
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) # use Adam for better accuracy

    dataloader = batch_provider(inputs, targets, 2)

    val_results = np.genfromtxt('valClasses.csv',dtype=int).astype(np.float32)
    print(val_results.dtype)
    val_vectors = np.genfromtxt('valFeatures.csv',dtype=float, delimiter=",")

    for i in range(nb_epochs):
        correctly_classified = 0
        print("Processing epoch number: " + str(i))
        epochs_list.append(i)
        minibatches_iterations = 0
        losses_minibatches_sum = 0

        for x, t in batch_provider(inputs, targets, batch_size):
            #print(f'x: {x}, t: {t}')
            sigmoid_result = model.forward(x)
            loss = F.binary_cross_entropy(model.forward(x), t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses_minibatches_sum = losses_minibatches_sum + loss.item() # accumulate losses of minibatches
            minibatches_iterations += 1

        # convert float64 val_dataset to float32 because of prob_class_1
        val_dataset_inputs = val_vectors.astype(np.float32)
        val_predictions = model.prob_class_1(val_dataset_inputs)
        # threshold predictions to 0 or 1, compare with real value and compute accuracy
        k = np.where(val_predictions<0.5,0,np.where(val_predictions>=0.5,1,val_predictions))

        for target, prediction in zip(val_results, k):
            if (target == prediction):
                correctly_classified += 1

        accuracy = (correctly_classified / np.shape(val_results)[0])
        # deepcopy if model has better accuracy then max accuracy from list
        if (len(accuracies) > 0):
            if (accuracy > max(accuracies)):
                best_model = copy.deepcopy(model)

        accuracies.append(accuracy) # save accuracy of val dataset of processed epoch to list
        loss_average = losses_minibatches_sum / minibatches_iterations # compute average loss for epoch
        losses.append(loss_average) # save loss of processed epoch to list

        print("Accuracy for this epoch: " + str(accuracy))

    #print(accuracies)
    return best_model, losses, accuracies, epochs_list

trained_results = np.genfromtxt('trainedClasses.csv',dtype=int).astype(np.float32)
print(trained_results.shape)
trained_vectors = np.genfromtxt('trainedImprovements.csv',dtype=float, delimiter=",").astype(np.float32)
print(trained_vectors.shape)
print(trained_vectors.dtype)
best_model_full, losses_full, accuracies_full, epochs_list_full = train_all_fea_llr(1000, 0.01, 2, trained_vectors, trained_results)
print("Max accuracy of model: " + str(max(accuracies_full)))

figure = plt.figure(figsize=(10, 10))
performance_plot = figure.add_subplot(2,1,1)
performance_plot.plot(epochs_list_full, accuracies_full, color = "orchid", label="accuracy development")
performance_plot.set_title('All Features Logistic Regression Performance', fontsize=10)
performance_plot.set_xlabel('Count of epochs', fontsize=8, horizontalalignment='right', x=1.0)
performance_plot.legend(prop={'size': 10})

performance_plot2 = figure.add_subplot(2,1,2)
performance_plot2.plot(epochs_list_full, losses_full, color = "indigo", label="loss development")
performance_plot2.set_xlabel('Count of epochs', fontsize=8, horizontalalignment='right', x=1.0)
performance_plot2.legend(prop={'size': 10})

plt.show()
