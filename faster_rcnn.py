# models/faster_rcnn.py

import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

def get_model(num_classes, freeze_backbone=False, train_backbone_layers=None):
    # Backbone
    backbone = resnet_fpn_backbone('resnet50', pretrained=True)

    # Freeze all if requested
    if freeze_backbone:
        for name, param in backbone.body.named_parameters():
            param.requires_grad = False

    # Unfreeze only specific layers
    if train_backbone_layers:
        for name, param in backbone.body.named_parameters():
            if any(name.startswith(layer) for layer in train_backbone_layers):
                param.requires_grad = True

    # Create model
    model = FasterRCNN(backbone, num_classes=num_classes)
    return model
