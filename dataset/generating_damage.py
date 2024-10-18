from PIL import Image, ImageDraw, JpegImagePlugin
from io import BytesIO
import numpy as np
from datasets import load_dataset

def generate_square_damage(image: JpegImagePlugin.JpegImageFile, square_side=64) -> JpegImagePlugin.JpegImageFile:
    image = image.copy()
    
    draw = ImageDraw.Draw(image)

    x = np.random.randint(0, image.size[0] - square_side)
    y = np.random.randint(0, image.size[1] - square_side)

    square_coords = [(x, y), (x + square_side, y + square_side)]

    draw.rectangle(square_coords, fill=(255, 255, 255))

    buffer = BytesIO()
    image.save(buffer, format='JPEG', quality=100)
    buffer.seek(0)  # Cofnięcie wskaźnika na początek bufora

    return Image.open(buffer)

def add_damaged_images(batch):
    batch['image_square_damage'] = [generate_square_damage(image, 32) for image in batch['image']]
    return batch

if __name__ == '__main__':
    ds = load_dataset('Artificio/WikiArt_Full', split='train')
    ds_with_damaged_images = ds.map(add_damaged_images, batched=True, num_proc=4)
    ds_with_damaged_images.save_to_disk('WikiArt_damaged')
