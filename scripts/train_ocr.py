def main():
    import os
    import torch
    import torch.nn as nn
    import matplotlib.pyplot as plt

    from tqdm import tqdm

    from plate_recognition.data import OCR_train_dataset, OCR_valid_dataset, OCR_train_loader, OCR_valid_loader, PATH_TO_SAVE_MODEL
    from plate_recognition.early_stopping import EarlyStopping
    from plate_recognition.models import Model, OCR_str_info

    plt.style.use('dark_background')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("Введите кол-во эпох и коэффициент значимости улучшения (по умолчанию = 0.001)")

    COAF = float(input())
    EPOCHS = int(input())
    save_epoch = 0
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    lr_list = []
    best_loss = None

    # Объявляем и указываем параметры нейронной сети
    OCR_Detection_NN = Model(8).to(device)
    early_stop = EarlyStopping(mode='min', patience=10)

    loss_regression = nn.SmoothL1Loss()
    opt = torch.optim.AdamW(OCR_Detection_NN.parameters(), lr=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=2, threshold=0.001, cooldown=1)

    print(f"Начало обучения с кол-вом эпох - {EPOCHS}...")

    # Цикл обучения нейронной сети

    for epoch in range(EPOCHS):

        # Тренировка модели
        OCR_Detection_NN.train()
        running_train_loss = []
        cnt = 0
        true_answers = 0
        train_loop = tqdm(OCR_train_loader, leave=False)
        for x, targets in train_loop:
            x = x.to(device)
            targets = targets.to(device)

            # Прямой проход и расчет ошибки модели
            prediction = OCR_Detection_NN(x)
            loss = loss_regression(prediction, targets)

            # Обратный проход
            opt.zero_grad()
            loss.backward()
            # Шаг оптимизации
            opt.step()

            running_train_loss.append(loss.item())
            mean_train_loss = sum(running_train_loss) / len(running_train_loss)

            true_answers += (torch.abs(prediction - targets)<= 5).all(dim=1).sum().item()

            train_loop.set_description(f"Epoch [{epoch + 1}/{EPOCHS}], train_loss={mean_train_loss:.4f}")

        # Расчет значения метрики
        running_train_acc = true_answers / len(OCR_train_dataset)

        # Сохранение значения функции потерь и метрики
        train_loss.append(mean_train_loss)
        train_acc.append(running_train_acc)

        # Validation
        OCR_Detection_NN.eval()
        with torch.no_grad():
            running_val_loss = []
            true_answers = 0
            for x, targets in OCR_valid_loader:
                x = x.to(device)
                targets = targets.to(device)

                prediction = OCR_Detection_NN(x)
                loss = loss_regression(prediction, targets)

                running_val_loss.append(loss.item())
                mean_val_loss = sum(running_val_loss) / len(running_val_loss)

                true_answers += (torch.abs(prediction - targets)<= 5).all(dim=1).sum().item()

            # Расчет значения метрики
            running_val_acc = true_answers / len(OCR_valid_dataset)

            # Сохранение значения функции потерь и метрики
            val_loss.append(mean_val_loss)
            val_acc.append(running_val_acc)

            is_first_checkpoint = best_loss is None
            if is_first_checkpoint:
                best_loss = mean_val_loss

            if is_first_checkpoint or mean_val_loss < best_loss - best_loss * COAF:
                best_loss = mean_val_loss

                checkpoint = {
                    'info': OCR_str_info,
                    'state_model': OCR_Detection_NN.state_dict(),
                    'state_opt': opt.state_dict(),
                    'state_lr_scheduler': lr_scheduler.state_dict(),

                    'loss': {
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'best_loss': best_loss
                    },

                    'metric': {
                        'train_acc': train_acc,
                        'val_acc': val_acc,
                    },

                    'lr': lr_list,

                    'epoch': {
                        'EPOCH': EPOCHS,
                        'save_epoch': save_epoch
                    }
                }

                torch.save(checkpoint, os.path.join(PATH_TO_SAVE_MODEL, f"Dict_OCR_Detection_NN_{save_epoch}.pt"))

                print(
                    f"Модель с эпохой - {epoch}, со значением функции потерь на валидации - {mean_val_loss:.4f} сохранена.",
                    end='\n\n')

            if early_stop(mean_val_loss):
                torch.save(checkpoint, os.path.join(PATH_TO_SAVE_MODEL, f"Dict_OCR_Detection_NN_{save_epoch}.pt"))
                print(f"Финальное сохранение модели с эпохой - {epoch}")
                print(f"\033[31mОбучение остановлено на {epoch} эпохе. \033[0m")
                break

        lr_scheduler.step(mean_val_loss)
        lr = lr_scheduler._last_lr[0]
        lr_list.append(lr)

        save_epoch += 1

        print(
            f"Epoch [{epoch}/{EPOCHS}], train_loss = {mean_train_loss:.4f}, train_acc = {running_train_acc:.4f}, val_loss = {mean_val_loss:.4f}, val_acc = {running_val_acc:.4f}, lr={lr:.4f}")

    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.legend(['loss_train', 'loss_val'])
    plt.show()

    plt.plot(train_acc)
    plt.plot(val_acc)
    plt.legend(['acc_train', 'acc_val'])
    plt.show()


if __name__ == '__main__':
    main()
