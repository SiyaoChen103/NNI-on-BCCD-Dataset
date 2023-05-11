# NNI-on-BCCD-Dataset
Neural Network Intelligence on Blood Cell Images Classification.
We learned to apply NNI on the BCCD dataset and use neural architecture search to find an optimal neural network for classificating the images into 4 blood cell types.
We also trained other neural network(ResNet50) on the dataset in order to compare their performances with the optimal model picked by NNI.

## Data Preparation
We downloaded the dataset from Kaggle: https://www.kaggle.com/datasets/paultimothymooney/blood-cells

Here is the plot of distribution of blood cells of 4 target classes for train and test sets, which is very balanced.

![distributions](https://github.com/SiyaoChen103/NNI-on-BCCD-Dataset/blob/main/data_distribution.png?raw=true)

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

The image below displays part of the training process

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

![nni_acc](https://github.com/SiyaoChen103/NNI-on-BCCD-Dataset/blob/main/nni_model_acc.png?raw=true)

## ResNet50 Model on BCCD Dataset
We also applied ResNet50 Model on the same dataset to compare the accuracy results, referencing a sample code on Kaggle. 
The source code of this section is stored in the ResNet50 folder.

The test accuracy of the ResNet50 model was 0.6238, which is higher than the model selected by nni.

Below is a classfication report generated.

![resnet50_acc](https://github.com/SiyaoChen103/NNI-on-BCCD-Dataset/blob/main/ResNet50_acc.png?raw=true)

## Observation
ResNet50 has much better performance than the model selected by NNI Dartstrainer. 

However, we did not define a broad search space for the CNN used in neural architecture search as we kept encountering technical difficulties, and the basic CNN structure was simple, whereras the ResNet50 model was a more complicated architecture with stacking of residual blocks.
So it was not too surprising that the selected model did not have a better performance than the ResNet50 model.

## Acknowledgements
dataset: https://www.kaggle.com/code/st1ckman/cnn-blood-cells

nni documentation: https://nni.readthedocs.io/en/stable/

nni DartsTrainer Sourcecode: https://github.com/microsoft/nni/blob/master/nni/retiarii/oneshot/pytorch/darts.py

nni DartsTrainer sample: https://nni.readthedocs.io/en/stable/deprecated/oneshot_legacy.html

resnet50 sample: https://www.kaggle.com/code/siddhantojha17/blood-cell-classification-using-resnet50




