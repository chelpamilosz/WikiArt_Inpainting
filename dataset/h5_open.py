import h5py
import numpy as np

# Funkcja do odczytu danych z pliku HDF5
def load_dataset_from_h5(file_name):
    with h5py.File(file_name, 'r') as f:
        # Odczytywanie kolumn tekstowych i konwersja z bajtów na stringi (utf-8)
        titles = [title.decode('utf-8') for title in f['title']]
        artists = [artist.decode('utf-8') for artist in f['artist']]
        dates = [date.decode('utf-8') for date in f['date']]
        genres = [genre.decode('utf-8') for genre in f['genre']]
        styles = [style.decode('utf-8') for style in f['style']]
        descriptions = [description.decode('utf-8') for description in f['description']]
        filenames = [filename.decode('utf-8') for filename in f['filename']]

        # Odczytywanie cech resnet50
        resnet50_non_robust_feats = np.array(f['resnet50_non_robust_feats'])
        resnet50_robust_feats = np.array(f['resnet50_robust_feats'])

        # Odczytywanie PCA embeddings
        embeddings_pca512 = np.array(f['embeddings_pca512'])

        # Odczytywanie obrazów
        images = np.array(f['images'])

        return {
            'title': titles,
            'artist': artists,
            'date': dates,
            'genre': genres,
            'style': styles,
            'description': descriptions,
            'filename': filenames,
            'resnet50_non_robust_feats': resnet50_non_robust_feats,
            'resnet50_robust_feats': resnet50_robust_feats,
            'embeddings_pca512': embeddings_pca512,
            'images': images
        }

# Odczytaj dataset z pliku HDF5
dataset = load_dataset_from_h5('dataset.h5')

# Przykład wyświetlenia tytułów
print(dataset['title'])
