import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from tqdm import tqdm
from PIL import Image, ImageDraw

from torchvision import transforms

from plate_recognition.data import (
    OCR_test_loader,
    OCR_train_loader,
    OCR_valid_loader,
    REC_train_loader,
    REC_valid_loader,
    REC_test_loader,
    PATH_TO_SAVE_MODEL,
    PATH_TO_TEST_DATASET,
    PATH_TO_TEST_DATASET1,
    REC_transform
)

from plate_recognition.models import (
    Model as OCR_Model,
    Rec_Model,
    OCR_str_info,
    REC_str_info
)

from plate_recognition.early_stopping import EarlyStopping


device = 'cuda' if torch.cuda.is_available() else 'cpu'

plt.style.use('dark_background')

COAF = 0.05

# Пути к последним сохранённым моделям
OCR_CHECKPOINT = os.path.join(
    PATH_TO_SAVE_MODEL,
    'Dict_OCR_Detection_NN_5.pt'
)

REC_CHECKPOINT = os.path.join(
    PATH_TO_SAVE_MODEL,
    'Dict_REC_Detection_NN_21.pt'
)

# Алфавит для распознавания номеров
ALPHABET = '0123456789ABEKMHOPCTYX'


# --------------------------------------------------
# ЗАГРУЗКА МОДЕЛИ И CHECKPOINT
# --------------------------------------------------

def load_model(model, checkpoint_path):

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f'Checkpoint не найден: {checkpoint_path}'
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    model.load_state_dict(checkpoint['state_model'])

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=0.001
    )

    opt.load_state_dict(checkpoint['state_opt'])

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt
    )

    lr_scheduler.load_state_dict(
        checkpoint['state_lr_scheduler']
    )

    print(f'\nМодель загружена: {checkpoint_path}')

    return model, checkpoint, opt, lr_scheduler


# --------------------------------------------------
# МЕТРИКА
# --------------------------------------------------

def calculate_metric(prediction, targets, model_type):

    if model_type == 'OCR':

        # Detection: правильность всех 8 координат
        correct = (
            torch.abs(prediction - targets) <= 5
        ).all(dim=1).sum().item()

        total = targets.size(0)

    else:

        # REC: точность распознавания отдельных символов
        predicted_chars = prediction.argmax(dim=2)
        actual_chars = targets.argmax(dim=2)

        # Учитываем только реальные символы.
        # Пустые строки One-Hot матрицы не учитываем.
        valid_positions = targets.sum(dim=2) > 0

        correct = (
            (predicted_chars == actual_chars) &
            valid_positions
        ).sum().item()

        total = valid_positions.sum().item()

    return correct, total


# --------------------------------------------------
# ОБУЧЕНИЕ
# --------------------------------------------------

def train_model(
    model,
    checkpoint,
    opt,
    lr_scheduler,
    train_loader,
    valid_loader,
    model_type,
    model_info
):

    loss_regression = nn.SmoothL1Loss()
    early_stop = EarlyStopping(mode='min', patience=10)

    print('Вы выбрали дообучение модели')
    EPOCHS = int(input('Введите количество эпох: '))

    save_epoch = checkpoint['epoch']['save_epoch'] + 1

    train_loss = checkpoint['loss']['train_loss']
    train_acc = checkpoint['metric']['train_acc']

    val_loss = checkpoint['loss']['val_loss']
    val_acc = checkpoint['metric']['val_acc']

    lr_list = checkpoint['lr']

    best_loss = checkpoint['loss']['best_loss']

    print(f'Начало обучения. Количество эпох: {EPOCHS}')

    for epoch in range(EPOCHS):

        # ------------------------------
        # TRAIN
        # ------------------------------

        model.train()

        running_train_loss = []
        true_answers = 0
        total_answers = 0

        train_loop = tqdm(train_loader, leave=False)

        for x, targets in train_loop:

            x = x.to(device)
            targets = targets.to(device)

            prediction = model(x)

            loss = loss_regression(
                prediction,
                targets
            )

            opt.zero_grad()
            loss.backward()
            opt.step()

            running_train_loss.append(loss.item())

            correct, total = calculate_metric(
                prediction,
                targets,
                model_type
            )

            true_answers += correct
            total_answers += total

            mean_train_loss = sum(running_train_loss) / len(running_train_loss)

            train_loop.set_description(
                f'Epoch [{epoch + 1}/{EPOCHS}], '
                f'train_loss={mean_train_loss:.4f}'
            )

        running_train_acc = (
            true_answers / total_answers
            if total_answers > 0 else 0
        )

        train_loss.append(mean_train_loss)
        train_acc.append(running_train_acc)

        # ------------------------------
        # VALIDATION
        # ------------------------------

        model.eval()

        running_val_loss = []
        true_answers = 0
        total_answers = 0

        with torch.inference_mode():

            for x, targets in valid_loader:

                x = x.to(device)
                targets = targets.to(device)

                prediction = model(x)

                loss = loss_regression(
                    prediction,
                    targets
                )

                running_val_loss.append(loss.item())

                correct, total = calculate_metric(
                    prediction,
                    targets,
                    model_type
                )

                true_answers += correct
                total_answers += total

        mean_val_loss = sum(running_val_loss) / len(running_val_loss)

        running_val_acc = (
            true_answers / total_answers
            if total_answers > 0 else 0
        )

        val_loss.append(mean_val_loss)
        val_acc.append(running_val_acc)

        # ------------------------------
        # СОХРАНЕНИЕ ЛУЧШЕЙ МОДЕЛИ
        # ------------------------------

        if (
            best_loss is None or
            mean_val_loss < best_loss * (1 - COAF)
        ):

            best_loss = mean_val_loss

            checkpoint_save = {
                'info': model_info,

                'state_model': model.state_dict(),
                'state_opt': opt.state_dict(),
                'state_lr_scheduler': lr_scheduler.state_dict(),

                'loss': {
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'best_loss': best_loss
                },

                'metric': {
                    'train_acc': train_acc,
                    'val_acc': val_acc
                },

                'lr': lr_list,

                'epoch': {
                    'EPOCH': EPOCHS,
                    'save_epoch': save_epoch
                }
            }

            save_path = os.path.join(
                PATH_TO_SAVE_MODEL,
                f'Dict_{model_type}_Detection_NN_{save_epoch}.pt'
            )

            torch.save(checkpoint_save, save_path)

            print(
                f'\nЛучшая модель сохранена. '
                f'Epoch: {epoch + 1}, '
                f'Val Loss: {mean_val_loss:.4f}'
            )

        # ------------------------------
        # EARLY STOPPING
        # ------------------------------

        if early_stop(mean_val_loss):

            print(
                f'\nОбучение остановлено на эпохе {epoch + 1}'
            )

            break

        # ------------------------------
        # SCHEDULER
        # ------------------------------

        lr_scheduler.step(mean_val_loss)

        lr = opt.param_groups[0]['lr']
        lr_list.append(lr)

        print(
            f'Epoch [{epoch + 1}/{EPOCHS}], '
            f'train_loss={mean_train_loss:.4f}, '
            f'train_acc={running_train_acc:.4f}, '
            f'val_loss={mean_val_loss:.4f}, '
            f'val_acc={running_val_acc:.4f}, '
            f'lr={lr:.6f}'
        )

        save_epoch += 1

    # ------------------------------
    # ГРАФИКИ
    # ------------------------------

    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.legend(['loss_train', 'loss_val'])
    plt.title('Loss')
    plt.show()

    plt.plot(train_acc)
    plt.plot(val_acc)
    plt.legend(['acc_train', 'acc_val'])
    plt.title('Accuracy')
    plt.show()


# --------------------------------------------------
# ТЕСТИРОВАНИЕ
# --------------------------------------------------

def test_model(model, test_loader, model_type):

    loss_regression = nn.SmoothL1Loss()

    model.eval()

    test_loss = []
    true_answers = 0
    total_answers = 0

    with torch.inference_mode():

        for x, targets in test_loader:

            x = x.to(device)
            targets = targets.to(device)

            prediction = model(x)

            loss = loss_regression(
                prediction,
                targets
            )

            test_loss.append(loss.item())

            correct, total = calculate_metric(
                prediction,
                targets,
                model_type
            )

            true_answers += correct
            total_answers += total

    mean_test_loss = sum(test_loss) / len(test_loss)

    test_acc = (
        true_answers / total_answers
        if total_answers > 0 else 0
    )

    print(
        f'Test Loss = {mean_test_loss:.4f}, '
        f'Accuracy = {test_acc:.4f}'
    )


# --------------------------------------------------
# ПРОСМОТР ФОТОГРАФИЙ DETECTION
# --------------------------------------------------

def view_ocr_images(model):

    image_files = [
        file for file in os.listdir(PATH_TO_TEST_DATASET)
        if file.lower().endswith(
            ('.jpg', '.jpeg', '.png', '.bmp')
        )
    ]

    image_files.sort()

    if not image_files:
        print('В папке нет фотографий!')
        return

    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor()
    ])

    model.eval()

    index = 0

    while True:

        image_path = os.path.join(
            PATH_TO_TEST_DATASET,
            image_files[index]
        )

        image = Image.open(image_path).convert('RGB')

        original_width, original_height = image.size

        x = transform(image).unsqueeze(0).to(device)

        with torch.inference_mode():
            prediction = model(x)

        coords = prediction[0].cpu().tolist()

        print(f'\nФотография: {image_files[index]}')
        print(f'Предсказания: {coords}')

        scale_x = original_width / 640
        scale_y = original_height / 640

        coords = [
            coords[i] * (scale_x if i % 2 == 0 else scale_y)
            for i in range(8)
        ]

        draw = ImageDraw.Draw(image)

        draw.polygon(
            coords,
            outline='green',
            width=4
        )

        image.show()

        print('\n[n] Следующая | [p] Предыдущая | [q] Выход')

        command = input('Команда: ').lower().strip()

        if command == 'n':
            index = (index + 1) % len(image_files)

        elif command == 'p':
            index = (index - 1) % len(image_files)

        elif command == 'q':
            break

        else:
            print('Неизвестная команда!')

    print('Просмотр завершён.')


# --------------------------------------------------
# ПРОСМОТР ФОТОГРАФИЙ REC
# --------------------------------------------------

def view_rec_images(model):

    image_files = [
        file for file in os.listdir(PATH_TO_TEST_DATASET1)
        if file.lower().endswith(
            ('.jpg', '.jpeg', '.png', '.bmp')
        )
    ]

    image_files.sort()

    if not image_files:
        print('В папке нет фотографий!')
        return

    model.eval()

    index = 0

    while True:

        image_path = os.path.join(
            PATH_TO_TEST_DATASET1,
            image_files[index]
        )

        image = Image.open(image_path).convert('RGB')

        x = REC_transform(image).unsqueeze(0).to(device)

        with torch.inference_mode():
            prediction = model(x)

        prediction = prediction[0].cpu()

        # Получаем наиболее вероятный символ
        result = ''

        for row in prediction:

            char_index = torch.argmax(row).item()

            result += ALPHABET[char_index]

        print(f'\nФотография: {image_files[index]}')
        print(f'Распознанный номер: {result}')

        image.show()

        print('\n[n] Следующая | [p] Предыдущая | [q] Выход')

        command = input('Команда: ').lower().strip()

        if command == 'n':
            index = (index + 1) % len(image_files)

        elif command == 'p':
            index = (index - 1) % len(image_files)

        elif command == 'q':
            break

        else:
            print('Неизвестная команда!')

    print('Просмотр завершён.')


def main():
    # ==================================================
    # ВЫБОР МОДЕЛИ
    # ==================================================

    print('\nВыберите модель:')
    print('1 - OCR_Model (Detection)')
    print('2 - REC_Model (Распознавание)')

    var_m = int(input('Ваш выбор: '))


    # ==================================================
    # OCR / DETECTION
    # ==================================================

    if var_m == 1:

        Model_1 = OCR_Model(8).to(device)

        Model_1, Model_state, opt, lr_scheduler = load_model(
            Model_1,
            OCR_CHECKPOINT
        )

        train_loader = OCR_train_loader
        valid_loader = OCR_valid_loader
        test_loader = OCR_test_loader

        model_type = 'OCR'
        model_info = OCR_str_info

        print('\nВыберите действие:')
        print('1. Дообучение модели')
        print('2. Тестирование модели')
        print('3. Проверка на фотографиях')
        print('4. Сохранение весов модели')

        variable = int(input('Ваш выбор: '))

        if variable == 1:

            train_model(
                Model_1,
                Model_state,
                opt,
                lr_scheduler,
                train_loader,
                valid_loader,
                model_type,
                model_info
            )

        elif variable == 2:

            test_model(
                Model_1,
                test_loader,
                model_type
            )

        elif variable == 3:

            view_ocr_images(Model_1)

        elif variable == 4:

            save_path = os.path.join(
                PATH_TO_SAVE_MODEL,
                'OCR_Detection_NN.pt'
            )

            torch.save(Model_1.state_dict(), save_path)

            print(f'Веса сохранены: {save_path}')


    # ==================================================
    # REC / RECOGNITION
    # ==================================================

    elif var_m == 2:

        Model_1 = Rec_Model(9 * 22).to(device)

        Model_1, Model_state, opt, lr_scheduler = load_model(
            Model_1,
            REC_CHECKPOINT
        )

        train_loader = REC_train_loader
        valid_loader = REC_valid_loader
        test_loader = REC_test_loader

        model_type = 'REC'
        model_info = REC_str_info

        print('\nВыберите действие:')
        print('1. Дообучение модели')
        print('2. Тестирование модели')
        print('3. Проверка распознавания на фотографиях')
        print('4. Сохранение весов модели')

        variable = int(input('Ваш выбор: '))

        if variable == 1:

            train_model(
                Model_1,
                Model_state,
                opt,
                lr_scheduler,
                train_loader,
                valid_loader,
                model_type,
                model_info
            )

        elif variable == 2:

            test_model(
                Model_1,
                test_loader,
                model_type
            )

        elif variable == 3:

            view_rec_images(Model_1)

        elif variable == 4:

            save_path = os.path.join(
                PATH_TO_SAVE_MODEL,
                'REC_Detection_NN.pt'
            )

            torch.save(Model_1.state_dict(), save_path)

            print(f'Веса сохранены: {save_path}')

    else:

        print('Неизвестный номер модели!')



if __name__ == '__main__':
    main()
