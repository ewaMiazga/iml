import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim # is this ok to import??

## MS2
    
class MLP(nn.Module):
    """
    An MLP network which does classification.

    It should not use any convolutional layers.
    """

    def __init__(self, input_size, n_classes):
        """
        Initialize the network.
        
        You can add arguments if you want, but WITH a default value, e.g.:
            __init__(self, input_size, n_classes, my_arg=32)
        
        Arguments:
            input_size (int): size of the input
            n_classes (int): number of classes to predict
        """
        super().__init__()
        ##
        ###
        #### WRITE YOUR CODE HERE! 
        ###
        ##
        
    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, D)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """
        ##
        ###
        #### WRITE YOUR CODE HERE! 
        ###
        ##
       # return preds


class CNN(nn.Module):
    """
    A CNN which does classification.

    It should use at least one convolutional layer.
    """

    def __init__(self, input_channels, n_classes, weight_decay=None):
        """
        Initialize the network.
        
        You can add arguments if you want, but WITH a default value, e.g.:
            __init__(self, input_channels, n_classes, my_arg=32)
        
        Arguments:
            input_channels (int): number of channels in the input
            n_classes (int): number of classes to predict
        """
        super().__init__()
        
         #LeNet
        self.conv2d1 = nn.Conv2d(input_channels, 6, kernel_size=5, padding=2)
        self.conv2d2 = nn.Conv2d(6, 16, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(16*8*8,120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_classes)
        
    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, Ch, H, W)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """
        x = F.relu(self.conv2d1(x))
        x = F.max_pool2d(x, kernel_size=2)
        
        x = F.relu(self.conv2d2(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = x.view(-1, 16*8*8)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        preds = self.fc3(x)
    
        return preds


class Trainer(object):
    """
    Trainer class for the deep networks.

    It will also serve as an interface between numpy and pytorch.
    """

    def __init__(self, model, lr, epochs, batch_size, weight_decay=None):
        """
        Initialize the trainer object for a given model.

        Arguments:
            model (nn.Module): the model to train
            lr (float): learning rate for the optimizer
            epochs (int): number of epochs of training
            batch_size (int): number of data points in each batch
        """
        self.lr = lr
        self.epochs = epochs
        self.model = model
        self.batch_size = batch_size
        self.weight_decay = weight_decay

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        if self.weight_decay is not None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def train_all(self, dataloader):
        """
        Fully train the model over the epochs. 
        
        In each epoch, it calls the functions "train_one_epoch". If you want to
        add something else at each epoch, you can do it here.

        Arguments:
            dataloader (DataLoader): dataloader for training data
        """
        for ep in range(self.epochs):
            self.train_one_epoch(dataloader)
            train_acc = self.calculate_accuracy(dataloader)
            print(f"Epoch {ep+1} - Training accuracy: {train_acc:.2f}%")

            ### WRITE YOUR CODE HERE if you want to do add else at each epoch

    def train_one_epoch(self, dataloader):
        """
        Train the model for ONE epoch.

        Should loop over the batches in the dataloader. (Recall the exercise session!)
        Don't forget to set your model to training mode, i.e., self.model.train()!

        Arguments:
            dataloader (DataLoader): dataloader for training data
        """
        self.model.train()

        for it, batch in enumerate(dataloader):
            x, y = batch
            logits = self.model(x)
            loss = self.criterion(logits, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def predict_torch(self, dataloader):
        """
        Predict the validation/test dataloader labels using the model.

        Hints:
            1. Don't forget to set your model to eval mode, i.e., self.model.eval()!
            2. You can use torch.no_grad() to turn off gradient computation, 
            which can save memory and speed up computation. Simply write:
                with torch.no_grad():
                    # Write your code here.

        Arguments:
            dataloader (DataLoader): dataloader for validation/test data
        Returns:
            pred_labels (torch.tensor): predicted labels of shape (N,),
                with N the number of data points in the validation/test data.
        """
        self.model.eval()
        pred_labels = []

        with torch.no_grad():
            for data in dataloader:
                inputs = data[0]
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                pred_labels.extend(predicted.tolist())

        pred_labels = torch.tensor(pred_labels)
        return pred_labels
    
    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        This serves as an interface between numpy and pytorch.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): regression target of shape (N,)
        Returns:
            pred_labels (array): target of shape (N,)
        """
        # First, prepare data for pytorch
        train_dataset = TensorDataset(torch.from_numpy(training_data).float(), 
                                      torch.from_numpy(training_labels))
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        self.train_all(train_dataloader)

        return self.predict(training_data)

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        This serves as an interface between numpy and pytorch.
        
        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        # First, prepare data for pytorch
        test_dataset = TensorDataset(torch.from_numpy(test_data).float())
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        pred_labels = self.predict_torch(test_dataloader)

        # We return the labels after transforming them into numpy array.
        return pred_labels.numpy()
    
    def calculate_accuracy(self, dataloader):
        """
        Calculate the accuracy of the model predictions on the given data loader.

        Arguments:
            dataloader (DataLoader): DataLoader for the dataset
        Returns:
            accuracy (float): Accuracy of the model predictions in percentage
        """
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data in dataloader:
                inputs, labels = data
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        accuracy = (correct / total) * 100
        return accuracy