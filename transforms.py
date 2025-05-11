# datasets/transforms.py
import torchvision.transforms as T

def get_transform(train):
    """
    Defines transformations for the dataset.
    Args:
        train (bool): Whether it's for training or validation.
    Returns:
        transform (torchvision.transforms.Compose): Transformation pipeline.
    """
    transform_list = []
    transform_list.append(T.ToTensor())
    
    if train:
        transform_list.append(T.RandomHorizontalFlip())  # Data Augmentation
    
    transform_list.append(T.Resize((800, 800)))  # Resize to a fixed size
    
    return T.Compose(transform_list)
