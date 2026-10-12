import torch.nn as nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


class SmallCNN(nn.Module):
    def __init__(self, classes):
        super().__init__()
        layers = []
        channels = [3, 16, 32, 64, 96]
        for incoming, outgoing in zip(channels[:-1], channels[1:]):
            layers.extend([nn.Conv2d(incoming, outgoing, 3, stride=2, padding=1, bias=False),
                           nn.BatchNorm2d(outgoing), nn.ReLU(inplace=True)])
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(96, classes))

    def forward(self, images):
        return self.classifier(self.features(images))



def mobilenet(classes, pretrained=False):
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, classes)
    return model


def unfreeze_features(model, blocks):
    if blocks < 0 or blocks > len(model.features) - 2:
        raise ValueError("Invalid number of bottleneck blocks to unfreeze")
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    if blocks:
        # Include the final projection as well as the requested bottlenecks.
        for layer in model.features[-(blocks + 1):]:
            for parameter in layer.parameters():
                parameter.requires_grad = True
