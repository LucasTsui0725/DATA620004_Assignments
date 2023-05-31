import torchvision
from torchvision.models.detection.fcos import FCOSClassificationHead


def create_model_FCOS(num_classes):
    
    model = torchvision.models.detection.fcos_resnet50_fpn(
        num_classes=num_classes,pretrained=False)

    return model
