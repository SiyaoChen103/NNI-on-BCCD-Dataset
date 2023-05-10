# NNI-on-BCCD-Dataset
Neural Network Intelligence on Blood Cell Images Classification.
We learned to apply NNI on the BCCD dataset and use neural architecture serach to find an optimal neural network for classificatio the images into 4 blood cell types.
We also trained other neural network on the dataset in order to compare their performances with the optimal model picked by NNI.

## Data Preparation
We downloaded the dataset from Kaggle: https://www.kaggle.com/datasets/paultimothymooney/blood-cells

## Environment setup
In Google Colab
```
!pip install nni
!pip install pytorch_lightning
```

## Search Space
We build a basic CNN model, and used simple search space focusing on following convolutional layers:
```
        self.conv1 = nn.LayerChoice([nn.Conv2d(3, 6, 3, padding=1), nn.Conv2d(3, 6, 5, padding=2)])
        self.conv2 = nn.LayerChoice([nn.Conv2d(6, 16, 3, padding=1), nn.Conv2d(6, 16, 5, padding=2)])
        self.skipconnect = nn.InputChoice(n_candidates=2)
```
## NNI implementation
Tutorial on NNI documentation: https://nni.readthedocs.io/en/stable/nas/exploration_strategy.html

Image below displays part of the training process
![exported](https://github.com/SiyaoChen103/NNI-on-BCCD-Dataset/blob/main/nni_process.png?raw=true)


## Exported Structure
![exported](https://github.com/SiyaoChen103/NNI-on-BCCD-Dataset/blob/main/exported.png?raw=true)

The result means that the following parameters inside the model are selected in the following three selections:
```
  #self.conv1 = nn.Conv2d(3, 6, 5, padding=2)
  #self.conv2 =nn.Conv2d(6, 16, 3, padding=1）
  #self.skipconnect = [0] (Skip connections are not used)
```
## Evaluate Chosen Model
We rebuilt the model with the selected architecture, and performed training and evaluating the model. The source code is stored in the NNI_model folder. 
We ran for 50 epochs, and the final test accuracy for the model was 0.4910.

## ResNet50 Model on BCCD Dataset
We also applied ResNet50 Model on the same dataset to compare the accuracy results. 
The source code of this section is stored in the ResNet50 folder.






