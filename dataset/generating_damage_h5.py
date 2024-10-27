import h5py
import numpy as np
from PIL import Image, ImageDraw
from datasets import load_from_disk
from time import time

def generate_square_damage(image: Image.Image, square_side=64) -> Image.Image:
    image = image.copy()  # Tworzymy kopię obrazu
    draw = ImageDraw.Draw(image)

    # Losowa pozycja kwadratu
    x = np.random.randint(0, image.size[0] - square_side)
    y = np.random.randint(0, image.size[1] - square_side)
    square_coords = [(x, y), (x + square_side, y + square_side)]

    # Rysowanie białego kwadratu
    draw.rectangle(square_coords, fill=(255, 255, 255))
    
    return image

if __name__ == '__main__':
    # Załaduj dane z dysku
    ds = load_from_disk(r'C:\vscodeProjects\WikiArt_Inpainting\WikiArt_damaged_100')
    
    start = time()
    images = ds['image']

    # Przetwarzanie obrazów: nakładanie uszkodzenia, konwersja na numpy i skalowanie do uint8
    image_arrays = [np.array(generate_square_damage(img).convert("RGB"), dtype=np.uint8) for img in images]

    # Połącz listę numpy array w jedną tablicę (B x H x W x C) i zamień oś na (B x C x H x W)
    image_array_stack = np.stack(image_arrays).transpose(0, 3, 1, 2)

    # Zapisz dane numpy do pliku .h5 jako uint8
    with h5py.File('damaged_images_tensor_data.h5', 'w') as h5f:
        h5f.create_dataset('image', data=image_array_stack, compression='gzip', compression_opts=9, dtype='uint8')

    end = time()

    print(f'{end - start} seconds')
    print("Obrazy z uszkodzeniami zostały zapisane do pliku damaged_images_tensor_data.h5 jako uint8")
