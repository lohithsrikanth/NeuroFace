import os
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader

def get_transform():
  train_transform = transforms.Compose([
      transforms.Resize((224, 224)),
      transforms.RandomHorizontalFlip(),
      transforms.ColorJitter(brightness=0.2, contrast=0.2),
      transforms.ToTensor(),
      transforms.Normalize(
         mean=[0.485, 0.456, 0.406],
         std=[0.229, 0.224, 0.225]
  )

  ])

  eval_transform = transforms.Compose([
      transforms.Resize((224, 224)),
      transforms.ToTensor(),
      transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
     )
  ])
  return train_transform, eval_transform

def get_dataloaders(data_root:str, batch_size=32, num_workers=2):
  train_transform, eval_transform = get_transform()

  train_dir = os.path.join(data_root, "train")
  val_dir   = os.path.join(data_root, "val")
  test_dir  = os.path.join(data_root, "test")

  train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
  val_dataset   = datasets.ImageFolder(val_dir, transform=eval_transform)
  test_dataset  = datasets.ImageFolder(test_dir, transform=eval_transform)

  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
  val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
  test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

  return train_loader, val_loader, test_loader, test_dataset, train_dataset.classes
