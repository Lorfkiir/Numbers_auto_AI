
import os

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from tqdm import tqdm

from plate_recognition.data import (
    REC_train_loader,
    REC_valid_loader,
    PATH_TO_SAVE_MODEL,
)

from plate_recognition.early_stopping import EarlyStopping
from plate_recognition.models import Rec_Model, REC_str_info


# ==================================================
# МЕТРИКИ
# ==================================================

def calculate_metrics(prediction, targets):
    """
    prediction: [batch, 9, 22]
    targets:    [batch, 9, 22]

    Возвращает:
        correct_chars - количество правильных символов
        total_chars - количество реальных символов
        correct_numbers - количество полностью правильных номеров
        total_numbers - количество номеров
    """

    predicted_chars = prediction.argmax(dim=2)
    actual_chars = targets.argmax(dim=2)

    # Маска реальных символов (исключаем пустые позиции)
    valid_positions = targets.sum(dim=2) > 0

    # Правильность отдельных символов
    correct_chars = (
        (predicted_chars == actual_chars) & valid_positions
    ).sum().item()

    total_chars = valid_positions.sum().item()

    # Правильность целых номеров
    correct_numbers = (
        ((predicted_chars == actual_chars) | ~valid_positions)
        .all(dim=1)
        .sum()
        .item()
    )

    total_numbers = targets.size(0)

    return (
        correct_chars,
        total_chars,
        correct_numbers,
        total_numbers
    )


# ==================================================
# ОБУЧЕНИЕ
# ==================================================

def main():

    plt.style.use('dark_background')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('Введите коэффициент улучшения и количество эпох:')

    COAF = float(input())
    EPOCHS = int(input())

    # История обучения
    save_epoch = 0

    train_loss = []
    val_loss = []

    train_char_acc = []
    val_char_acc = []

    train_exact_acc = []
    val_exact_acc = []

    lr_list = []

    best_loss = None

    # ==================================================
    # МОДЕЛЬ И ОПТИМИЗАТОР
    # ==================================================

    model = Rec_Model(9 * 22).to(device)

    early_stop = EarlyStopping(
        mode='min',
        patience=10
    )

    loss_regression = nn.CrossEntropyLoss()

    opt = torch.optim.Adam(
        model.parameters(),
        lr=0.0005
    )

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode='min',
        factor=0.5,
        patience=2,
        threshold=0.001,
        cooldown=1
    )

    print(f'\nУстройство: {device}')
    print(f'Начало обучения. Количество эпох: {EPOCHS}\n')

    # ==================================================
    # EPOCH LOOP
    # ==================================================

    for epoch in range(EPOCHS):

        # ==============================================
        # TRAIN
        # ==============================================

        model.train()

        running_train_loss = 0.0
        train_samples = 0

        train_correct_chars = 0
        train_total_chars = 0

        train_correct_numbers = 0
        train_total_numbers = 0

        train_loop = tqdm(
            REC_train_loader,
            leave=False
        )

        for x, targets in train_loop:

            x = x.to(device)
            targets = targets.to(device)

            # Forward
            prediction = model(x)

            loss = loss_regression(
                prediction,
                targets
            )

            # Backward
            opt.zero_grad()
            loss.backward()
            opt.step()

            # Loss
            batch_size = x.size(0)

            running_train_loss += loss.item() * batch_size
            train_samples += batch_size

            # Metrics
            with torch.no_grad():

                (
                    correct_chars,
                    total_chars,
                    correct_numbers,
                    total_numbers
                ) = calculate_metrics(prediction, targets)

            train_correct_chars += correct_chars
            train_total_chars += total_chars

            train_correct_numbers += correct_numbers
            train_total_numbers += total_numbers

            mean_train_loss = running_train_loss / train_samples

            train_loop.set_description(
                f'Epoch [{epoch + 1}/{EPOCHS}] '
                f'Loss: {mean_train_loss:.4f}'
            )

        # Итоговые метрики TRAIN
        mean_train_loss = running_train_loss / train_samples

        running_train_char_acc = (
            train_correct_chars / train_total_chars
            if train_total_chars > 0 else 0
        )

        running_train_exact_acc = (
            train_correct_numbers / train_total_numbers
            if train_total_numbers > 0 else 0
        )

        train_loss.append(mean_train_loss)

        train_char_acc.append(running_train_char_acc)
        train_exact_acc.append(running_train_exact_acc)

        # ==============================================
        # VALIDATION
        # ==============================================

        model.eval()

        running_val_loss = 0.0
        val_samples = 0

        val_correct_chars = 0
        val_total_chars = 0

        val_correct_numbers = 0
        val_total_numbers = 0

        with torch.inference_mode():

            for x, targets in REC_valid_loader:

                x = x.to(device)
                targets = targets.to(device)

                prediction = model(x)

                loss = loss_regression(
                    prediction,
                    targets
                )

                # Loss
                batch_size = x.size(0)

                running_val_loss += loss.item() * batch_size
                val_samples += batch_size

                # Metrics
                (
                    correct_chars,
                    total_chars,
                    correct_numbers,
                    total_numbers
                ) = calculate_metrics(prediction, targets)

                val_correct_chars += correct_chars
                val_total_chars += total_chars

                val_correct_numbers += correct_numbers
                val_total_numbers += total_numbers

        # Итоговые метрики VALIDATION
        mean_val_loss = running_val_loss / val_samples

        running_val_char_acc = (
            val_correct_chars / val_total_chars
            if val_total_chars > 0 else 0
        )

        running_val_exact_acc = (
            val_correct_numbers / val_total_numbers
            if val_total_numbers > 0 else 0
        )

        val_loss.append(mean_val_loss)

        val_char_acc.append(running_val_char_acc)
        val_exact_acc.append(running_val_exact_acc)

        # ==============================================
        # СОХРАНЕНИЕ ЛУЧШЕЙ МОДЕЛИ
        # ==============================================

        is_first_checkpoint = best_loss is None

        if (
            is_first_checkpoint
            or mean_val_loss < best_loss * (1 - COAF)
        ):

            best_loss = mean_val_loss

            checkpoint = {
                'info': REC_str_info,

                'state_model': model.state_dict(),
                'state_opt': opt.state_dict(),
                'state_lr_scheduler': lr_scheduler.state_dict(),

                'loss': {
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'best_loss': best_loss
                },

                'metric': {
                    'train_char_acc': train_char_acc,
                    'val_char_acc': val_char_acc,

                    'train_exact_acc': train_exact_acc,
                    'val_exact_acc': val_exact_acc,
                },

                'lr': lr_list,

                'epoch': {
                    'EPOCH': EPOCHS,
                    'save_epoch': save_epoch
                }
            }

            save_path = os.path.join(
                PATH_TO_SAVE_MODEL,
                f'Dict_REC_Detection_NN_{save_epoch}.pt'
            )

            torch.save(checkpoint, save_path)

            print(
                f'\nЛучшая модель сохранена. '
                f'Epoch: {epoch + 1}, '
                f'Val Loss: {mean_val_loss:.4f}'
            )

        # ==============================================
        # EARLY STOPPING
        # ==============================================

        if early_stop(mean_val_loss):

            print(
                f'\nОбучение остановлено на эпохе {epoch + 1}'
            )

            break

        # ==============================================
        # SCHEDULER
        # ==============================================

        lr_scheduler.step(mean_val_loss)

        lr = opt.param_groups[0]['lr']
        lr_list.append(lr)

        # ==============================================
        # ИНФОРМАЦИЯ ОБ ЭПОХЕ
        # ==============================================

        print(f'\nEpoch [{epoch + 1}/{EPOCHS}]')

        print(
            f'TRAIN: '
            f'Loss = {mean_train_loss:.4f} | '
            f'Char Acc = {running_train_char_acc:.2%} | '
            f'Exact Match = {running_train_exact_acc:.2%}'
        )

        print(
            f'VALID: '
            f'Loss = {mean_val_loss:.4f} | '
            f'Char Acc = {running_val_char_acc:.2%} | '
            f'Exact Match = {running_val_exact_acc:.2%}'
        )

        print(f'LR = {lr:.6f}')

        save_epoch += 1

    # ==================================================
    # ГРАФИКИ
    # ==================================================

    # Loss
    plt.figure(figsize=(10, 5))

    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Validation Loss')

    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(alpha=0.2)

    plt.show()

    # Character Accuracy
    plt.figure(figsize=(10, 5))

    plt.plot(
        train_char_acc,
        label='Train Character Accuracy'
    )

    plt.plot(
        val_char_acc,
        label='Validation Character Accuracy'
    )

    plt.title('Character Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(alpha=0.2)

    plt.show()

    # Exact Match Accuracy
    plt.figure(figsize=(10, 5))

    plt.plot(
        train_exact_acc,
        label='Train Exact Match'
    )

    plt.plot(
        val_exact_acc,
        label='Validation Exact Match'
    )

    plt.title('Exact Match Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(alpha=0.2)

    plt.show()


if __name__ == '__main__':
    main()
