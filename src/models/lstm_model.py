from __future__ import annotations

from copy import deepcopy
import logging
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


logger = logging.getLogger(__name__)


class GlobalLSTM(nn.Module):
    def __init__(self, hidden_size=64, num_layers=1, dropout=0.0):
        super().__init__()
        effective_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(3, hidden_size, num_layers=num_layers, dropout=effective_dropout, batch_first=True)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x):
        encoded, _ = self.lstm(x)
        return self.output(encoded[:, -1]).squeeze(-1)


def _wape(y, pred):
    return np.abs(y - pred).sum() / (np.abs(y).sum() + 1e-8)


def _loader(dataset, batch_size, shuffle, workers, pin_memory):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers, pin_memory=pin_memory,
                      persistent_workers=workers > 0)


def train_lstm(model, train_dataset, validation_dataset, epochs=20, batch_size=256,
               learning_rate=1e-3, patience=3, device=None, dataloader_workers=0,
               pin_memory=True):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.SmoothL1Loss()
    use_pin_memory = bool(pin_memory and str(device).startswith("cuda"))
    train_loader = _loader(train_dataset, batch_size, True, dataloader_workers, use_pin_memory)
    val_loader = (_loader(validation_dataset, batch_size, False, dataloader_workers, use_pin_memory)
                  if validation_dataset is not None else None)
    best, state, stale = float("inf"), None, 0
    logger.info("开始 LSTM 训练：epochs=%d, train_windows=%d, validation_windows=%s, device=%s",
                epochs, len(train_dataset),
                len(validation_dataset) if validation_dataset is not None else "关闭", device)
    best_epoch = epochs
    for epoch in range(epochs):
        model.train()
        for x, y, _, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x.to(device, non_blocking=use_pin_memory)),
                           y.to(device, non_blocking=use_pin_memory))
            loss.backward(); optimizer.step()
        if val_loader is None:
            logger.info("LSTM epoch %d/%d 完成", epoch + 1, epochs)
            continue
        pred, true = predict_lstm(model, val_loader, device)
        score = _wape(true, pred)
        logger.info("LSTM epoch %d/%d 完成，validation WAPE=%.6f", epoch + 1, epochs, score)
        if score < best:
            best, state, stale, best_epoch = score, deepcopy(model.state_dict()), 0, epoch + 1
        else:
            stale += 1
            if stale >= patience:
                logger.info("LSTM 提前停止：最佳 epoch=%d，最佳 WAPE=%.6f", best_epoch, best)
                break
    if state is not None:
        model.load_state_dict(state)
    model.best_epoch = best_epoch
    logger.info("LSTM 训练完成：使用 epoch=%d 的参数", best_epoch)
    return model


def predict_lstm(model, dataset_or_loader, device=None, dataloader_workers=0,
                 pin_memory=True):
    device = device or next(model.parameters()).device
    use_pin_memory = bool(pin_memory and str(device).startswith("cuda"))
    loader = (dataset_or_loader if isinstance(dataset_or_loader, DataLoader)
              else _loader(dataset_or_loader, 256, False, dataloader_workers, use_pin_memory))
    predictions, targets = [], []
    model.eval()
    with torch.no_grad():
        for x, y, _, _ in loader:
            predictions.append(model(x.to(device, non_blocking=use_pin_memory)).cpu().numpy())
            targets.append(y.numpy())
    return np.maximum(0, np.concatenate(predictions)), np.concatenate(targets)
