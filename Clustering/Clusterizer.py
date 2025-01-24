#Biblioteki i pakiety
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader

import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.utils import resample
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt
from matplotlib import colors

from tqdm import tqdm
from joblib import dump, load

class Clusterizer():
    def __init__(self, comet_logger):
        
        self.comet_logger = comet_logger

        self.train_features = None
        self.valid_features = None
        self.test_features = None
    
    def load_features(self, type, features_path):
        if type == 'train':
            self.train_features = np.load(features_path)
        elif type == 'valid':
            self.valid_features = np.load(features_path)
        elif type == 'test':
            self.test_features = np.load(features_path)
        else:
            raise ValueError('Invalid type provided. Use "train", "valid", or "test".')
        
    def draw_ebolow_method(self, k_start = 1, k_end = 10):

        features = self.train_features['features']  # Wyciągnięcie cech
        # Zakres liczby klastrów do przetestowania
        k_values = range(k_start, k_end)
        inertia = []

        # Obliczanie inercji dla różnych liczby klastrów
        for k in tqdm(k_values, desc="Testing number of clusters"):
            kmeans = KMeans(n_clusters=k, random_state=1234)
            kmeans.fit(features)  # Użycie wczytanych cech
            inertia.append(kmeans.inertia_)

        # Tworzenie figury i jej przypisanie
        fig = plt.figure(figsize=(15, 10))
        plt.plot(k_values, inertia, 'bo-', markersize=8)
        plt.xlabel('Liczba klastrów (k)', fontsize=14)
        plt.ylabel('Inercja', fontsize=14)
        plt.title('Metoda łokcia dla optymalnego k', fontsize=16)
        plt.grid()

        # Logowanie wykresu na Comet
        self.comet_logger.experiment.log_figure(fig)
        plt.close(fig)  # Zamknięcie figury

    def draw_calinski_harabasz_method(self, k_start=2, k_end=10):
        features = self.train_features['features']  # Wyciągnięcie cech
        k_values = range(k_start, k_end)  # Zakres liczby klastrów
        scores = []

        # Obliczanie Calinski-Harabasz Index dla różnych liczby klastrów
        for k in tqdm(k_values, desc="Testing number of clusters with Calinski-Harabasz"):
            kmeans = KMeans(n_clusters=k, random_state=1234)
            labels = kmeans.fit_predict(features)  # Klasteryzacja
            score = calinski_harabasz_score(features, labels)  # Obliczenie metryki
            scores.append(score)

        # Tworzenie figury i wykresu
        fig = plt.figure(figsize=(15, 10))
        plt.plot(k_values, scores, 'bo-', markersize=8)
        plt.xlabel('Liczba klastrów (k)', fontsize=14)
        plt.ylabel('Calinski-Harabasz Index', fontsize=14)
        plt.title('Metoda Calinskiego-Harabasza dla optymalnego k', fontsize=16)
        plt.grid()

        # Logowanie wykresu na Comet
        self.comet_logger.experiment.log_figure(fig)
        plt.close(fig)  # Zamknięcie figury

    def draw_silhouette_score(self, k_start=2, k_end=10, sample_fraction=1.0):
        features = self.train_features['features']  # Wyciągnięcie cech
        
        # Próbkowanie danych, jeśli sample_fraction jest mniejsze od 1.0
        if 0.0 < sample_fraction < 1.0:
            features_sampled = resample(features, n_samples=int(len(features) * sample_fraction), random_state=1234)
        else:
            features_sampled = features

        k_values = range(k_start, k_end)  # Zakres liczby klastrów
        scores = []

        # Obliczanie współczynnika sylwetki dla różnych liczby klastrów
        for k in tqdm(k_values, desc="Testing number of clusters with Silhouette Score"):
            kmeans = KMeans(n_clusters=k, random_state=1234)
            labels = kmeans.fit_predict(features_sampled)  # Klasteryzacja
            score = silhouette_score(features_sampled, labels)  # Obliczenie współczynnika sylwetki
            scores.append(score)

        # Tworzenie figury i wykresu
        fig = plt.figure(figsize=(15, 10))
        plt.plot(k_values, scores, 'bo-', markersize=8)
        plt.xlabel('Liczba klastrów (k)', fontsize=14)
        plt.ylabel('Silhouette Score', fontsize=14)
        plt.title('Metoda współczynnika sylwetki dla optymalnego k', fontsize=16)
        plt.grid()

        # Logowanie wykresu na Comet
        self.comet_logger.experiment.log_figure(fig)
        plt.close(fig)  # Zamknięcie figury

    def draw_spectral_method(self, k_start = 2, k_end = 10, sample_fraction = 0.1):

        features = self.train_features['features']  # Wyciągnięcie cech

        # Redukcja danych do 10% oryginalnego zbioru
        n_samples = int(features.shape[0] * sample_fraction)  # Liczba próbek po redukcji
        subset_indices = np.random.choice(features.shape[0], n_samples, replace=False)  # Losowe indeksy
        subset_features = features[subset_indices]  # Podzbiór danych

        # Zakres liczby klastrów do przetestowania
        k_values = range(k_start, k_end)  # Spectral clustering wymaga co najmniej 2 klastrów
        silhouette_scores = []

        # Przetwarzanie Spectral Clustering dla różnych liczby klastrów
        for k in tqdm(k_values, desc="Testing number of clusters"):
            spectral = SpectralClustering(n_clusters=k, affinity='nearest_neighbors', random_state=1234)
            labels = spectral.fit_predict(subset_features)  # Klasteryzacja na podzbiorze danych

            # Obliczanie wyniku silhouette (dla oceny jakości klasteryzacji)
            score = silhouette_score(subset_features, labels)
            silhouette_scores.append(score)

        # Wizualizacja metody silhouette
        fig = plt.figure(figsize=(15, 10))
        plt.plot(k_values, silhouette_scores, 'bo-', markersize=8)
        plt.xlabel('Liczba klastrów (k)', fontsize=14)
        plt.ylabel('Silhouette Score', fontsize=14)
        plt.title('Analiza Silhouette dla optymalnego k', fontsize=16)
        plt.grid()
        # Logowanie wykresu na Comet
        self.comet_logger.experiment.log_figure(fig)
        plt.close(fig)
    

    def generate_and_log_cluster_plot(self, cluster_counts):
        # Klasteryzacja
        print("Rozpoczynam klasteryzację...")
        with tqdm(total=1, desc="Klasteryzacja") as pbar:
            kmeans = KMeans(n_clusters=cluster_counts, random_state=1234)
            cluster_labels = kmeans.fit_predict(self.train_features['features'])
            pbar.update(1)

        # Ograniczenie liczby próbek dla t-SNE
        max_samples = 10000
        features = self.train_features['features']
        sample_indices = np.random.choice(features.shape[0], size=min(max_samples, features.shape[0]), replace=False)
        sampled_features = features[sample_indices]
        sampled_labels = cluster_labels[sample_indices]

        # Redukcja wymiarów za pomocą t-SNE
        print("Rozpoczynam redukcję wymiarów (t-SNE)...")
        with tqdm(total=1, desc="t-SNE") as pbar:
            tsne = TSNE(n_components=2, random_state=1234)
            reduced_features_tsne = tsne.fit_transform(sampled_features)
            pbar.update(1)

        # Redukcja wymiarów za pomocą UMAP
        print("Rozpoczynam redukcję wymiarów (UMAP)...")
        with tqdm(total=1, desc="UMAP") as pbar:
            umap_reducer = umap.UMAP(n_components=2, random_state=1234)
            reduced_features_umap = umap_reducer.fit_transform(features)
            pbar.update(1)

        # Kolory dla klastrów
        unique_labels = np.unique(cluster_labels)
        color_palette = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
        cluster_colors = {label: color_palette[i] for i, label in enumerate(unique_labels)}

        # Mapowanie kolorów do etykiet klastrów (ograniczone do t-SNE próbek)
        colors_mapped_tsne = [cluster_colors[label] for label in sampled_labels]
        colors_mapped_umap = [cluster_colors[label] for label in cluster_labels]

        # Tworzenie wykresu t-SNE
        print("Tworzę wykres t-SNE...")
        fig_tsne, ax_tsne = plt.subplots(figsize=(10, 10))
        ax_tsne.scatter(
            reduced_features_tsne[:, 0],
            reduced_features_tsne[:, 1],
            c=colors_mapped_tsne,
            s=10
        )
        ax_tsne.set_title(f'Wizualizacja ({cluster_counts}) klastrów z t-SNE', fontsize=16)
        ax_tsne.set_xlabel('t-SNE Component 1', fontsize=14)
        ax_tsne.set_ylabel('t-SNE Component 2', fontsize=14)

        # Logowanie wykresu t-SNE na Comet
        print("Loguję wykres t-SNE na Comet...")
        self.comet_logger.experiment.log_figure(fig_tsne)

        plt.close(fig_tsne)

        # Tworzenie wykresu UMAP
        print("Tworzę wykres UMAP...")
        fig_umap, ax_umap = plt.subplots(figsize=(10, 10))
        ax_umap.scatter(
            reduced_features_umap[:, 0],
            reduced_features_umap[:, 1],
            c=colors_mapped_umap,
            s=10
        )
        ax_umap.set_title(f'Wizualizacja ({cluster_counts}) klastrów z UMAP', fontsize=16)
        ax_umap.set_xlabel('UMAP Component 1', fontsize=14)
        ax_umap.set_ylabel('UMAP Component 2', fontsize=14)

        # Logowanie wykresu UMAP na Comet
        print("Loguję wykres UMAP na Comet...")
        self.comet_logger.experiment.log_figure(fig_umap)

        plt.close(fig_umap)

        print("Proces zakończony.")
    
    def clusterize(self, type, n_clusters, image_features=None):
        if type == 'train':
            # Klasteryzacja zbioru treningowego
            features = self.train_features['features']
            indices = self.train_features['indices']  # Pobranie istniejących indeksów
            kmeans = KMeans(n_clusters=n_clusters, random_state=1234)
            cluster_labels = kmeans.fit_predict(features)

            # Zapis modelu KMeans
            dump(kmeans, '/content/drive/MyDrive/Models/kmeans_model.pkl')
            print("Model KMeans zapisany w: /content/drive/MyDrive/Models/kmeans_model.pkl")

            # Zapis klastrów i indeksów do pliku
            save_file_name = f'/content/drive/MyDrive/{type}_clusters.npz'
            np.savez_compressed(save_file_name, cluster_labels=cluster_labels, indices=indices)
            print(f"Klastry i indeksy zapisane w: {save_file_name}")
            return save_file_name

        elif type == 'valid':
            # Klasteryzacja zbioru walidacyjnego
            features = self.valid_features['features']
            indices = self.valid_features['indices']  # Pobranie istniejących indeksów
            kmeans = load('/content/drive/MyDrive/Models/kmeans_model.pkl')
            print("Załadowano model KMeans z: /content/drive/MyDrive/Models/kmeans_model.pkl")

            cluster_labels = kmeans.predict(features)

            # Zapis klastrów i indeksów do pliku
            save_file_name = f'/content/drive/MyDrive/{type}_clusters.npz'
            np.savez_compressed(save_file_name, cluster_labels=cluster_labels, indices=indices)
            print(f"Klastry i indeksy zapisane w: {save_file_name}")
            return save_file_name

        elif type == 'test':
            # Klasteryzacja zbioru testowego
            if image_features is None:
                raise ValueError("Dla typu 'test' wymagane jest przekazanie cech obrazu (image_features).")

            features = image_features['features']
            kmeans = load('/content/drive/MyDrive/Models/kmeans_model.pkl')
            print("Załadowano model KMeans z: /content/drive/MyDrive/Models/kmeans_model.pkl")

            cluster_labels = kmeans.predict(features)
            return cluster_labels

        else:
            raise ValueError("Nieprawidłowy typ danych. Użyj 'train', 'valid' lub 'test'.")
        
    def count_samples_per_cluster(self, file_path: str):
        try:
            # Wczytanie pliku .npz
            data = np.load(file_path)
            cluster_labels = data['cluster_labels']  # Wczytanie etykiet klastrów

            # Zliczenie ilości próbek w każdym klastrze
            cluster_counts = {}
            for label in np.unique(cluster_labels):
                cluster_counts[label] = np.sum(cluster_labels == label)

            print(f"Zliczone próbki dla każdego klastra: {cluster_counts}")
            return cluster_counts

        except FileNotFoundError:
            print(f"Plik {file_path} nie został znaleziony.")
            return None
        except KeyError:
            print(f"Plik {file_path} nie zawiera klucza 'cluster_labels'.")
            return None

    def get_images_from_random_cluster(self, train_cluster_file, valid_cluster_file, train_dataset, valid_dataset):
        try:
            # Wczytanie klastrów
            train_clusters = np.load(train_cluster_file)['cluster_labels']
            valid_clusters = np.load(valid_cluster_file)['cluster_labels']

            # Losowanie klastra
            unique_clusters = np.unique(train_clusters)
            random_cluster = np.random.choice(unique_clusters)

            print(f"Wybrano klaster: {random_cluster}")

            # Znalezienie indeksów dla wybranego klastra
            train_indices = np.where(train_clusters == random_cluster)[0]
            valid_indices = np.where(valid_clusters == random_cluster)[0]

            # Jeśli klaster ma mniej niż 4 obrazy w zbiorze, zgłoszenie błędu
            if len(train_indices) < 4 or len(valid_indices) < 4:
                raise ValueError(f"Wybrany klaster {random_cluster} ma za mało obrazów w jednym ze zbiorów.")

            # Losowe wybranie 4 indeksów z każdego zbioru
            selected_train_indices = np.random.choice(train_indices, 4, replace=False)
            selected_valid_indices = np.random.choice(valid_indices, 4, replace=False)

            # Pobranie obrazów
            train_images = [train_dataset[i][0] for i in selected_train_indices]
            valid_images = [valid_dataset[i][0] for i in selected_valid_indices]

            return train_images, valid_images, random_cluster

        except FileNotFoundError as e:
            print(f"Błąd: {e}")
            return None, None, None
        except KeyError as e:
            print(f"Błąd w pliku klastrów: {e}")
            return None, None, None
        except ValueError as e:
            print(f"Błąd: {e}")
            return None, None, None