from typing import Tuple
from torchvision import transforms

def get_train_transforms(image_size: int) -> transforms.Compose:
    """
    Conservative augmentation to avoid distorting diagnostic patterns:
        - Small rotations
        - Small translations / scale changes
        - Mild color jitter
        - No horizontal flip by default (to preserve left-right symmetry)
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomApply(
            [transforms.ColorJitter(brightness=0.1, contrast=0.1)], p=0.5
        ),
        transforms.RandomAffine(
            degrees=5,
            translate=(0.02, 0.02),
            scale=(0.95, 1.05)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def get_eval_transforms(image_size: int) -> transforms.Compose:
    """
    No augmentation, only resize & normalization.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])