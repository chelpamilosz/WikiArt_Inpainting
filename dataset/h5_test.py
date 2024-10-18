import h5py
import numpy as np

from datasets import load_dataset

from PIL import JpegImagePlugin, Image, ImageDraw
from io import BytesIO


ds = load_dataset('Artificio/WikiArt_Full', split='train')
ds_test = ds.select(range(500))


def generate_square_damage(image: JpegImagePlugin.JpegImageFile, square_side=64) -> JpegImagePlugin.JpegImageFile:
    draw = ImageDraw.Draw(image)

    x = np.random.randint(0, image.size[0] - square_side)
    y = np.random.randint(0, image.size[1] - square_side)

    square_coords = [(x, y), (x + square_side, y + square_side)]

    draw.rectangle(square_coords, fill=(255, 255, 255))

    buffer = BytesIO()
    image.save(buffer, format='JPEG', quality=100)
    buffer.seek(0)  # Cofnięcie wskaźnika na początek bufora

    # Załadowanie obrazu jako PngImageFile
    return Image.open(buffer)

# Funkcja konwertująca dataset do HDF5
def save_dataset_to_h5(ds, file_name):
    with h5py.File(file_name, 'w') as f:
        # Konwersja kolumn tekstowych na format UTF-8 i zapisanie w HDF5
        f.create_dataset('title', data=np.array([title.encode('utf-8') for title in ds['title']], dtype='S'))
        f.create_dataset('artist', data=np.array([artist.encode('utf-8') for artist in ds['artist']], dtype='S'))
        f.create_dataset('date', data=np.array([date.encode('utf-8') for date in ds['date']], dtype='S'))
        f.create_dataset('genre', data=np.array([genre.encode('utf-8') for genre in ds['genre']], dtype='S'))
        f.create_dataset('style', data=np.array([style.encode('utf-8') for style in ds['style']], dtype='S'))

        # Zapisywanie cech resnet50 (może być robust i non-robust)
        f.create_dataset('resnet50_non_robust_feats', data=np.array(ds['resnet50_non_robust_feats']))
        f.create_dataset('resnet50_robust_feats', data=np.array(ds['resnet50_robust_feats']))

        # Zapisywanie PCA embeddings (o zmiennej długości)
        f.create_dataset('embeddings_pca512', data=np.array(ds['embeddings_pca512']))

        # Zapisywanie obrazów (możesz je przekonwertować na tablice np. uint8)
        images = [np.array(image.convert('RGB')) for image in ds['image']]  # konwersja obrazów na RGB
        f.create_dataset('image', data=np.array(images))

        images_damaged_square = [np.array(generate_square_damage(image)) for image in ds['image']]
        f.create_dataset('image_damage_square', data=np.array(images_damaged_square))

# Zapisz dataset do pliku HDF5
save_dataset_to_h5(ds_test, 'dataset.h5')
