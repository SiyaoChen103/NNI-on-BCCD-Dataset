from torchvision import transforms
import torchvision

transformer = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = torchvision.datasets.ImageFolder(
        'blood-cell/dataset2-master/dataset2-master/images/TRAIN',
         transform=transformer)

test_dataset = torchvision.datasets.ImageFolder(
  'blood-cell/dataset2-master/dataset2-master/images/TEST/',
   transform=transformer
)
