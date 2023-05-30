# iml


## PCA:
- Kmeans:

python main2.py --data "data\dataset_HASYv2" --method kmeans --K 20                                                        | Accuracy: 77.283% | Time: 9.279160976409912

python main2.py --data "data\dataset_HASYv2" --method kmeans --K 20 --use_pca --pca_d 200                                  | Accuracy: 76.112% | Time: 0.6464433670043945

- Logistic regression:

python main2.py --data "data/dataset_HASYv2" --method logistic_regression --lr 1e-4 --max_iters 3000                       | Accuracy: 89.461% | Time: 18.433822870254517

python main2.py --data "data/dataset_HASYv2" --method logistic_regression --lr 1e-4 --max_iters 3000 --use_pca --pca_d 200 | Accuracy: 91.335% | Time: 7.751373529434204

- SVM:

python main2.py --data "data/dataset_HASYv2" --method svm --svm_c 10. --svm_kernel rbf --svm_gamma 0.001                   | Accuracy: 93.443% | Time: 3.989854574203491

python main2.py --data "data/dataset_HASYv2" --method svm --svm_c 10. --svm_kernel rbf --svm_gamma 0.001 --use_pca --pca_d 200 | Accuracy: 94.145% | Time: 0.9929873943328857

We clearly see that using PCA, allows us to improve time performance of the classifiers. Accuracy is not always inceasing. This vary depends on the model we are using.
