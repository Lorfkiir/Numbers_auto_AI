"""Пакетная обрезка номеров по координатам OCR-модели."""
import os
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from plate_recognition.config import PATH_TO_SAVE_MODEL, PROJECT_ROOT
from plate_recognition.models import Model


def main():
    input_dir = Path(os.getenv("CROP_INPUT_DIR", PROJECT_ROOT / "data" / "to_crop"))
    output_dir = Path(os.getenv("CROP_OUTPUT_DIR", PROJECT_ROOT / "data" / "cropped"))
    checkpoint_path = Path(os.getenv(
        "OCR_WEIGHTS",
        Path(PATH_TO_SAVE_MODEL) / "OCR_Detection_NN.pt",
    ))

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Папка с исходными изображениями не найдена: {input_dir}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Файл весов OCR-модели не найден: {checkpoint_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Model(8).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
    ])

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_paths = sorted(
        path for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in image_extensions
    )

    if not image_paths:
        print(f"В папке нет изображений: {input_dir}")
        return

    for image_path in image_paths:
        with Image.open(image_path) as source_image:
            image = source_image.convert("RGB")
            resized_image = image.resize((640, 640))
            tensor = transform(resized_image).unsqueeze(0).to(device)

            with torch.inference_mode():
                prediction = model(tensor)

            coords = prediction[0].cpu().tolist()
            crop_coords = coords[:2] + coords[4:6]
            cropped_image = resized_image.crop(crop_coords)
            cropped_image.save(output_dir / image_path.name)

        print(f"Сохранено: {output_dir / image_path.name}")


if __name__ == "__main__":
    main()
