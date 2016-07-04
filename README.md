# dlcv05

## Work in the UPC's Server

Enter the server: 
```
$ ssh -p 2241 dlcv05@147.83.91.181
password = dlcv2016
 ```
Activate Virtual Environment (Inside the Repository Folder)
```
source keras_env/bin/activate
```
Run a file
```
$ python asd.py
```

## Project
[Slides] (http://imatge-upc.github.io/telecombcn-2016-dlcv/slides/D1P-kickoff.pdf)

### Task 1: Architecture
Build your own network to solve a classification task.
### Task 2: Training
Study the impact in performance of:
- Data augmentation.
- Sizes of the training batches.
- Batch normalization

Draw your training and validation curves. 

Overfitting:
- Force an overfitting problem.
- Investigate if regularization (eg. drop out) reduces/solves it.

### Task 3: Visualization
Visualize filter responses:
- From your own network.
- From a pre-trained network.

t-SNE

Off-the-shelf AlexNet:
- Visualize local classification over a small set of images.

### Task 4: Transfer Learning

Train a network over CIFAR-10 and fine-tune over Terrassa Buildings 900.

Off-the-shelf convnet:
- Freeze weight in all layers but the last one, and replace it with a softmax to solve Terrassa Buildings 900.

### Task 5: Open Project


## Useful git commands

https://confluence.atlassian.com/bitbucketserver/basic-git-commands-776639767.html

## VirtualEnv + Keras installing guide for MacOS (Own Computer)

https://docs.google.com/document/d/1ovoac2bPKQSESzFCLdEc4-KSsKNHd15J4n6QGudpMlg/edit?usp=sharing

