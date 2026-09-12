import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from raw_sequence_utils import (
    NUM_OBJECTS,
    RawWindowDataset,
    evaluate_model,
    prepare_raw_datasets,
    set_seed,
    train_model,
)


class SEBlock1D(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class SEBasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.se = SEBlock1D(planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class RawMultiScaleSEGraRes(nn.Module):
    def __init__(self, block, num_classes=25):
        super().__init__()
        self.inplanes = 64
        self.conv1_small = nn.Conv1d(36, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv1_large = nn.Conv1d(36, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks=1)
        self.layer2 = self._make_layer(block, 128, blocks=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x_s = self.conv1_small(x)
        x_l = self.conv1_large(x)
        x = torch.cat([x_s, x_l], dim=1)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.fc(x)


def main():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    X_train, y_train, X_val, y_val, X_test, y_test = prepare_raw_datasets()
    train_loader = DataLoader(RawWindowDataset(X_train, y_train, layout="conv"), batch_size=32, shuffle=True)
    val_loader = DataLoader(RawWindowDataset(X_val, y_val, layout="conv"), batch_size=32)
    test_loader = DataLoader(RawWindowDataset(X_test, y_test, layout="conv"), batch_size=32)

    model_name = "MS+SE-GraRes"
    model = RawMultiScaleSEGraRes(SEBasicBlock1D, num_classes=NUM_OBJECTS).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    num_epochs = 30
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs, model_name
    )
    evaluate_model(model, test_loader, device, model_name, train_losses, val_losses, num_epochs)


if __name__ == "__main__":
    main()
