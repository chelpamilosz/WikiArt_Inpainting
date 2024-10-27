import h5py
import numpy as np
from datasets import load_from_disk
from time import time

if __name__ == '__main__':
    # Załaduj dane z dysku
    ds = load_from_disk(r'C:\vscodeProjects\WikiArt_Inpainting\WikiArt_damaged_100')
    
    start = time()
    # Zakładam, że twoja kolumna z obrazami nazywa się 'image'
    images = ds['image']

    # Przetwarzanie obrazów: konwersja na numpy array i skalowanie do uint8
    image_arrays = [np.array(img.convert("RGB"), dtype=np.uint8) for img in images]  # Konwersja obrazów na numpy i uint8

    # Połącz listę numpy array w jedną tablicę (B x H x W x C) i zamień oś na (B x C x H x W)
    image_array_stack = np.stack(image_arrays).transpose(0, 3, 1, 2)

    # Zapisz dane numpy do pliku .h5 jako uint8
    with h5py.File('images_tensor_data.h5', 'w') as h5f:
        h5f.create_dataset('image', data=image_array_stack, compression='gzip', compression_opts=9, dtype='uint8')

    end = time()

    print(f'{end - start} seconds')
    print("Obrazy zostały zapisane do pliku images_tensor_data.h5 jako uint8")
