import h5py
import numpy as np
from PIL import Image, ImageDraw
from datasets import load_from_disk
from time import time
from tqdm import tqdm

def generate_square_damage(image: Image.Image, square_side=64) -> Image.Image:
    image = image.copy()
    draw = ImageDraw.Draw(image)

    x = np.random.randint(0, image.size[0] - square_side)
    y = np.random.randint(0, image.size[1] - square_side)
    square_coords = [(x, y), (x + square_side, y + square_side)]

    draw.rectangle(square_coords, fill=(255, 255, 255))
    
    return image

if __name__ == '__main__':
    ds = load_from_disk(r'C:\vscodeProjects\WikiArt_Inpainting\WikiArt_damaged_100')
    
    print('ds loaded')

    start = time()
    num_images = len(ds)
    batch_size = num_images

    # sample_image = generate_square_damage(ds[0]['image'], square_side=32).convert("RGB")
    sample_image = ds[0]['image'].convert('RGB')
    image_shape = (3, sample_image.height, sample_image.width)

    print('sample image done')

    with h5py.File('images_tensor_data_100.h5', 'w') as h5f:
        dataset = h5f.create_dataset('image', shape=(num_images, *image_shape), 
                                     dtype='uint8', compression='gzip', compression_opts=9)

        print('h5 created')

        for i in tqdm(range(0, num_images, batch_size), desc='Batch', position=0):
            batch_images = ds[i:i + batch_size]['image']

            batch_data = [
                # np.array(generate_square_damage(img, square_side=32).convert("RGB"), dtype=np.uint8).transpose(2, 0, 1)
                np.array(img.convert('RGB'), dtype=np.uint8).transpose(2, 0, 1)
                for img in tqdm(batch_images, desc='Batch progress', position=1, leave=False)
            ]

            dataset[i:i + len(batch_data)] = batch_data

    end = time()
    print(f'{end - start} seconds')
    print('Done')
