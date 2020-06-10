import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from .logger import get_logger
from .timer import get_time_budget
from utils.callbacks import EarlyStopping

logger = get_logger("DEBUG")


def get_accuracy(y_hat, indices, data):
    accuracy = accuracy_score(
        data.y[indices].cpu().numpy(),
        y_hat[indices].argmax(axis=1).reshape(-1))
    return accuracy


def torch_train(
        data, dataset, model, optimizer, loss_func,
        epochs=512, batch_size=32, patience=5,
        clip_grad=0,
        min_epochs=1,
        valid_indices=None, all_data=False,
        use_adaptive_topK=False, model_topK=None,
        time_budget=None
):
    early_stopping_cb = EarlyStopping(patience=patience, min_epochs=min_epochs, use_adaptive_topK=use_adaptive_topK)
    early_stopping_cb.topK_list = model_topK
    # on epoch begin
    with tqdm(total=epochs) as t:
        try:
            for i in range(epochs):
                data_loader = DataLoader(dataset.resample(), batch_size=batch_size, shuffle=True)
                model.train()
                epoch_loss = 0
                for indices in data_loader:
                    # on batch begin
                    optimizer.zero_grad()
                    y_hat = model(data)
                    loss = loss_func(y_hat[indices].squeeze(), data.y[indices].squeeze())
                    loss.backward()
                    if clip_grad > 0:
                        for p in model.parameters():
                            nn.utils.clip_grad_norm_(p, clip_grad)
                    optimizer.step()
                    # on batch end
                    epoch_loss += loss.item()

                model.eval()
                with torch.no_grad():
                    y_hat = model(data)
                    y_hat = y_hat.cpu().numpy()

                # on epoch end
                if valid_indices is not None:
                    valid_acc = get_accuracy(y_hat, valid_indices, data)
                else:
                    valid_acc = get_accuracy(y_hat, data.train_indices, data)

                t.set_postfix(
                    Epoch=f"{i: 03,d}",
                    loss=f"{epoch_loss: 0.5f}",
                    acc=f"{valid_acc: 0.5f}",
                    patience=f"{early_stopping_cb.wait: 03,d}/{early_stopping_cb.patience}"
                )
                t.update(1)

                if early_stopping_cb.on_epoch_end(i, valid_acc, epoch_loss, y_hat):
                    break
                try:
                    time_budget.check()
                except Exception as e:
                    print(e)
                    return early_stopping_cb.topK_list, -early_stopping_cb.best, "time_exceed"
        except RuntimeError as exception:
            if "out of memory" in str(exception):
                logger.info("we met cuda out of memory")
                return early_stopping_cb.topK_list, -early_stopping_cb.best, "oom"
            else:
                raise exception
    return early_stopping_cb.topK_list, -early_stopping_cb.best, None
