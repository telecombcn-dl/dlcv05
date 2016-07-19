# DLCV05

## Project
[Project Slides] (http://imatge-upc.github.io/telecombcn-2016-dlcv/slides/D1P-kickoff.pdf)


### Task 1: Architecture
Build your own network to solve a classification task.

Script mnist_cnn.py:
- Options added at the beginning of the script
- We can save and load trained models
- We have the value of the loss & accuracy at each epoch
- Save total time computed


Script cifar10_cnn.py: 
- Options added at the beginning of the script
- We can save and load trained models
- We have the value of the loss & accuracy at each epoch
- Save total time computed

Script mnist_cnn_3layers.py:
- Custom architecture proposed

If you are saving the model be careful when setting the paths and the name not to overwrite!


### Task 2: Training

Objectives: 

Study the impact in performance of:
- Data augmentation.
- Sizes of the training batches.
- Batch normalization

Overfitting:
- Force an overfitting problem.
- Investigate if regularization (eg. drop out) reduces/solves it.

### Task 3: Visualization

Objective: 
Visualize filter responses
- There is the code to visualize the value of the weight as well as the output of every filter in our custom architecture on mnist database. 

### Task 4: Transfer Learning

(Experimental code ... not working properly when fine tuning) 

Train a network over CIFAR-10 and fine-tune over Terrassa Buildings 900.

Off the Shelf VGG-16

- Freeze weight in all layers but the last one, and replace it with a softmax to solve Terrassa Buildings 900.



### Task 5: Open Project

Adquire knwoledge and insights about what is happening inside the deepdream network. 
- Deepdream modify the images it is given as an input enhancing some features depending on the layer we choose to boost. 
- Lower layers focus on simpler features (edges, orientation, shapes)
- Higher layers focus on concrete objects that have been seen during training
