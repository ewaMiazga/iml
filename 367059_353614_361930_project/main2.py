import argparse

import numpy as np
from torchinfo import summary
import time

from src.data import load_data
from src.methods.dummy_methods import DummyClassifier
from src.methods.pca import PCA
from src.methods.deep_network import MLP, CNN, Trainer
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, get_n_classes
from src.methods.kmeans import KMeans
from src.methods.logistic_regression import LogisticRegression
from src.methods.svm import SVM



def main(args):
    """
    The main function of the script. Do not hesitate to play with it
    and add your own code, visualization, prints, etc!

    Arguments:
        args (Namespace): arguments that were parsed from the command line (see at the end 
                          of this file). Their value can be accessed as "args.argument".
    """
    ## 1. First, we load our data and flatten the images into vectors
    xtrain, xtest, ytrain, ytest = load_data(args.data)
    xtrain = xtrain.reshape(xtrain.shape[0], -1)
    xtest = xtest.reshape(xtest.shape[0], -1)


    ## 2. Then we must prepare it. This is were you can create a validation set,
    #  normalize, add bias, etc.

    # Make a validation set
    if not args.test:
        index_shuffled = np.random.permutation(len(xtrain)) # shuffle indices 
        xtrain, ytrain = xtrain[index_shuffled], ytrain[index_shuffled] # 
        size_val = int(0.2*len(xtrain)) # ratio for validation: 20% validation set and 80% training set
        xtest, ytest = xtrain[:size_val], ytrain[:size_val] #overwriting xtest 
        xtrain, ytrain = xtrain[size_val:], ytrain[size_val:]
    
    ### WRITE YOUR CODE HERE to do any other data processing
    xtrain_means = xtrain.mean(0,keepdims=True)
    xtrain_stds  = xtrain.std(0,keepdims=True)
    xtest_means  = xtest.mean(0,keepdims=True)
    xtest_stds  = xtest.std(0,keepdims=True)
    xtrain_normalized = normalize_fn(xtrain, xtrain_means, xtrain_stds)
    xtest_normalized = normalize_fn(xtest, xtest_means, xtest_stds) # normalize xtest with the same parameters as xtrain


    # Dimensionality reduction (MS2)
    if args.use_pca:
        print("Using PCA")
        pca_obj = PCA(d=args.pca_d)
        pca_obj.find_principal_components(xtest_normalized)
        
        xtrain_normalized = pca_obj.reduce_dimension(xtrain_normalized)
        xtest_normalized = pca_obj.reduce_dimension(xtest_normalized)
        ### WRITE YOUR CODE HERE: use the PCA object to reduce the dimensionality of the data


    ## 3. Initialize the method you want to use.

    # Neural Networks (MS2)
    if args.method == "nn":
        print("Using deep network")

        # Prepare the model (and data) for Pytorch
        # Note: you might need to reshape the image data depending on the network you use!
        number_classes = get_n_classes(ytrain)
        if args.nn_type == "mlp":
            model = MLP(
                input_size=xtrain_normalized.shape[1],  # Pass the input size
                n_classes=number_classes,
                hidden_dim1=256,  # Set the desired hidden layer dimensions
                hidden_dim2=128,
            )

        elif args.nn_type == "cnn":
            xtrain_normalized = xtrain_normalized.reshape(-1, 1, 32, 32)
            xtest_normalized = xtest_normalized.reshape(-1, 1, 32, 32)
            model = CNN(input_channels=1, n_classes=number_classes, weight_decay=args.weight_decay) #should use get_n_classes to generalize here!!
        summary(model)

        # Trainer object
        method_obj = Trainer(model, lr=args.lr, epochs=args.max_iters, batch_size=args.nn_batch_size)
     
    # Follow the "DummyClassifier" example for your methods (MS1)
    elif args.method == "dummy_classifier":
        method_obj =  DummyClassifier(arg1=1, arg2=2)

    elif args.method == "kmeans":
        method_obj = KMeans(
            K=args.K, 
            max_iters=args.max_iters
            )

    elif args.method == "logistic_regression":
        method_obj = LogisticRegression(
            lr=args.lr, 
            max_iters=args.max_iters
        )

    elif args.method == "svm":
        method_obj = SVM(
            C=args.svm_c,
            kernel=args.svm_kernel,
            gamma=args.svm_gamma
        )
        #method_obj.graph(xtrain, ytrain)
    

    ## 4. Train and evaluate the method

    # Fit (:=train) the method on the training data
    s1 = time.time()
    preds_train = method_obj.fit(xtrain_normalized, ytrain)
    
    # Predict on unseen data
    preds = method_obj.predict(xtest_normalized)

    s2 = time.time()
    print(f"Time analysis Overall: time = {s2-s1}")

    ## Report results: performance on train and valid/test sets
    acc = accuracy_fn(preds_train, ytrain)
    macrof1 = macrof1_fn(preds_train, ytrain)
    print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

    acc = accuracy_fn(preds, ytest)
    macrof1 = macrof1_fn(preds, ytest)
    print(f"Test set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

    ### WRITE YOUR CODE HERE if you want to add other outputs, visualization, etc.


if __name__ == '__main__':
    # Definition of the arguments that can be given through the command line (terminal).
    # If an argument is not given, it will take its default value as defined below.
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default="dataset_HASYv2", type=str, help="the path to wherever you put the data, if it's in the parent folder, you can use ../dataset_HASYv2")
    parser.add_argument('--method', default="dummy_classifier", type=str, help="dummy_classifier / kmeans / logistic_regression / svm / nn (MS2)")
    parser.add_argument('--K', type=int, default=10, help="number of clusters for K-Means")
    parser.add_argument('--lr', type=float, default=1e-5, help="learning rate for methods with learning rate")
    parser.add_argument('--max_iters', type=int, default=100, help="max iters for methods which are iterative")
    parser.add_argument('--test', action="store_true", help="train on whole training data and evaluate on the test data, otherwise use a validation set")
    parser.add_argument('--svm_c', type=float, default=1., help="Constant C in SVM method")
    parser.add_argument('--svm_kernel', default="linear", help="kernel in SVM method, can be 'linear' or 'rbf' or 'poly'(polynomial)")
    parser.add_argument('--svm_gamma', type=float, default=1., help="gamma prameter in rbf/polynomial SVM method")
    parser.add_argument('--svm_degree', type=int, default=1, help="degree in polynomial SVM method")
    parser.add_argument('--svm_coef0', type=float, default=0., help="coef0 in polynomial SVM method")

    ### WRITE YOUR CODE HERE: feel free to add more arguments here if you need!

    # MS2 arguments
    parser.add_argument('--use_pca', action="store_true", help="to enable PCA")
    parser.add_argument('--pca_d', type=int, default=200, help="output dimensionality after PCA")
    parser.add_argument('--nn_type', default="mlp", help="which network to use, can be 'mlp' or 'cnn'")
    parser.add_argument('--nn_batch_size', type=int, default=64, help="batch size for NN training")
    parser.add_argument('--weight_decay', type=float, default=None, help="L2 regularization parameter for CNN")
    # "args" will keep in memory the arguments and their values,
    # which can be accessed as "args.data", for example.
    args = parser.parse_args()
    main(args)
