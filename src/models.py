import torch.nn as nn


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


def make_model(name, classes, pretrained=False):
    if name == "cnn":
        return SmallCNN(classes)
    raise ValueError(f"Unknown model: {name}")
