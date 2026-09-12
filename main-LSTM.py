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


class RawLSTMLightModel(nn.Module):
    def __init__(self, input_size=36, hidden_size=16, num_layers=1, num_classes=25):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = False
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.dropout(out)
        return self.fc(out)


def main():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    X_train, y_train, X_val, y_val, X_test, y_test = prepare_raw_datasets()
    train_loader = DataLoader(RawWindowDataset(X_train, y_train, layout="lstm"), batch_size=32, shuffle=True)
    val_loader = DataLoader(RawWindowDataset(X_val, y_val, layout="lstm"), batch_size=32)
    test_loader = DataLoader(RawWindowDataset(X_test, y_test, layout="lstm"), batch_size=32)

    model_name = "LSTM-light"
    model = RawLSTMLightModel(input_size=36, num_classes=NUM_OBJECTS).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    num_epochs = 30
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs, model_name
    )
    evaluate_model(model, test_loader, device, model_name, train_losses, val_losses, num_epochs)


if __name__ == "__main__":
    main()
