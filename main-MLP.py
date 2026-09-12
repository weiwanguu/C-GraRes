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


class RawMLPLightModel(nn.Module):
    def __init__(self, input_size=36 * 110, num_classes=25):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.classifier(x)


def main():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    X_train, y_train, X_val, y_val, X_test, y_test = prepare_raw_datasets()
    train_loader = DataLoader(RawWindowDataset(X_train, y_train, layout="flat"), batch_size=32, shuffle=True)
    val_loader = DataLoader(RawWindowDataset(X_val, y_val, layout="flat"), batch_size=32)
    test_loader = DataLoader(RawWindowDataset(X_test, y_test, layout="flat"), batch_size=32)

    model_name = "MLP-light"
    model = RawMLPLightModel(num_classes=NUM_OBJECTS).to(device)
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
