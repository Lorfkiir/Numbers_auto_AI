import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from tqdm import tqdm

from Dataloader import test_dataset, test_loader, train_dataset, train_loader, valid_dataset, valid_loader, \
    PATH_TO_SAVE_MODEL, PATH_TO_TEST_DATASET
from Model import Model, OCR_str_info
from Earlystop import EarlyStopping
from PIL import Image, ImageDraw
from torchvision import transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'
plt.style.use('dark_background')

COAF = 0.05

Model = Model(8).to(device)
Model_state = torch.load(f'{PATH_TO_SAVE_MODEL}\\Dict_OCR_Detection_NN_26.pt')

loss_regression = nn.MSELoss()
opt = torch.optim.Adam(Model.parameters(), lr=0.001)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt)
early_stop = EarlyStopping()

Model.load_state_dict(Model_state['state_model'])
opt.load_state_dict(Model_state['state_opt'])
lr_scheduler.load_state_dict(Model_state['state_lr_scheduler'])

print("Выберите действие: ")
print("1. Дообучение модели")
print("2. Тестирование модели")
print("3. Запуск модели и проверка на фотографиях")

variable = int(input())

if variable == 1:

    print("Вы выбрали дообучение модели")
    print("Введите общее кол-во эпох:")
    EPOCHS = int(input())

    save_epoch = Model_state['epoch']['save_epoch']
    train_loss = Model_state['loss']['train_loss']
    train_acc = Model_state['metric']['train_acc']
    val_loss = Model_state['loss']['val_loss']
    val_acc = Model_state['metric']['val_acc']
    lr_list = Model_state['lr']
    best_loss = Model_state['loss']['best_loss']

    print(f"Начало обучения с кол-вом эпох - {EPOCHS}...")

    for epoch in range(save_epoch, EPOCHS):

        # Тренировка модели
        Model.train()
        running_train_loss = []
        true_answers = 0
        train_loop = tqdm(train_loader, leave=False)
        for x, targets in train_loop:
            x = x.to(device)
            targets = targets.to(device)

            # Прямой проход и расчет ошибки модели
            prediction = Model(x)
            loss = loss_regression(prediction, targets)

            # Обратный проход
            opt.zero_grad()
            loss.backward()
            # Шаг оптимизации
            opt.step()

            running_train_loss.append(loss.item())
            mean_train_loss = sum(running_train_loss) / len(running_train_loss)

            true_answers += (torch.abs(prediction - targets) <= 10).all(dim=1).sum().item()

            train_loop.set_description(f"Epoch [{epoch + 1}/{EPOCHS}], train_loss={mean_train_loss:.4f}")

        # Расчет значения метрики
        running_train_acc = true_answers / len(train_dataset)

        # Сохранение значения функции потерь и метрики
        train_loss.append(mean_train_loss)
        train_acc.append(running_train_acc)

        # Validation
        Model.eval()
        with torch.no_grad():
            running_val_loss = []
            true_answers = 0
            for x, targets in valid_loader:
                x = x.to(device)
                targets = targets.to(device)

                prediction = Model(x)
                loss = loss_regression(prediction, targets)

                running_val_loss.append(loss.item())
                mean_val_loss = sum(running_val_loss) / len(running_val_loss)

                true_answers += (torch.abs(prediction - targets) <= 10).all(dim=1).sum().item()

            # Расчет значения метрики
            running_val_acc = true_answers / len(valid_dataset)

            # Сохранение значения функции потерь и метрики
            val_loss.append(mean_val_loss)
            val_acc.append(running_val_acc)

            if best_loss is None:
                best_loss = mean_val_loss

            if mean_val_loss < best_loss - best_loss * COAF:
                best_loss = mean_val_loss

                save_epoch = epoch + 1
                checkpoint = {
                    'info': OCR_str_info,
                    'state_model': Model.state_dict(),
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

                torch.save(checkpoint, f"{PATH_TO_SAVE_MODEL}\\Dict_OCR_Detection_NN_{save_epoch}.pt")

                print(
                    f"Модель с эпохой - {epoch}, со значением функции потерь на валидации - {mean_val_loss:.4f} сохранена.",
                    end='\n\n')

            if early_stop(mean_val_loss):
                torch.save(Model.state_dict(), f'OCR_Detection_NN.pt')
                print(f"\033[31mОбучение остановлено на {epoch} эпохе. \033[0m")
                break

        lr_scheduler.step(mean_val_loss)
        lr = lr_scheduler._last_lr[0]
        lr_list.append(lr)

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

if variable == 2:

    print("Вы выбрали тестирование модели")

    Model.eval()

    with torch.inference_mode():
        test_loss = []
        true_answers = 0

        for x, targets in test_loader:
            x = x.to(device)
            targets = targets.to(device)

            prediction = Model(x)
            loss = loss_regression(prediction, targets)

            test_loss.append(loss.item())

            true_answers += (
                (torch.abs(prediction - targets) <= 10)
                .all(dim=1).sum().item()
            )

        mean_test_loss = sum(test_loss) / len(test_loss)
        test_acc = true_answers / len(test_dataset)

    print(f"Test Loss = {mean_test_loss:.4f}, Accuracy = {test_acc:.4f}.")

if variable == 3:

    # Получаем список фотографий
    image_files = [
        file for file in os.listdir(PATH_TO_TEST_DATASET)
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
    ]

    image_files.sort()

    if not image_files:
        print("В папке нет фотографий!")
        exit()

    # Подготовка изображений
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor()
    ])

    Model.eval()

    index = 0

    while True:

        image_path = os.path.join(PATH_TO_TEST_DATASET, image_files[index])

        # Загружаем фотографию
        image = Image.open(image_path).convert("RGB")

        original_width, original_height = image.size

        x = transform(image).unsqueeze(0).to(device)

        # Предсказание модели
        with torch.inference_mode():
            prediction = Model(x)

        coords = prediction[0].cpu().tolist()

        print(f"\nФотография: {image_files[index]}")
        print(f"Предсказания: {coords}")

        # Масштабируем координаты
        scale_x = original_width / 640
        scale_y = original_height / 640

        coords = [
            coords[i] * (scale_x if i % 2 == 0 else scale_y)
            for i in range(8)
        ]

        # Рисуем прямоугольники
        draw = ImageDraw.Draw(image)

        draw.polygon(coords[:8], outline="green", width=4)

        # Показываем фотографию
        image.show()

        # Управление через терминал
        print("\n[n] Следующая | [p] Предыдущая | [q] Выход")

        command = input("Команда: ").lower().strip()

        if command == 'n':
            index = (index + 1) % len(image_files)

        elif command == 'p':
            index = (index - 1) % len(image_files)

        elif command == 'q':
            break

        else:
            print("Неизвестная команда!")

    print("Просмотр завершён.")

if variable == 4:
    torch.save(Model.state_dict(), f"{PATH_TO_SAVE_MODEL}\\OCR_Detection_NN_.pt")
