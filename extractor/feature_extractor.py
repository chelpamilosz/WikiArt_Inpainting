import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from joblib import dump, load
import matplotlib.pyplot as plt
import os

from unet import UNetLightning

class FeatureExtractor():
    def __init__(self, model : UNetLightning, device, comet_logger, save_dir : str):
        
        self.model = model
        self.device = device
        self.comet_logger = comet_logger

        self.repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Domyślny Callback do zapisu najlepszych modeli
        self.checkpoint_callback = ModelCheckpoint(
            dirpath=save_dir,
            filename='unet-{epoch:02d}-{val_loss:.4f}',
            save_top_k=1,
            mode='min',
            monitor='val_loss'
        )
        # Domyślny Early Stopping Callback
        self.early_stopping_callback = EarlyStopping(
            monitor='val_loss',
            min_delta=0.00,
            patience=6,
            verbose=True,
            mode='min',
            check_on_train_epoch_end=False
        )

        # Domyślny Trainer
        self.trainer = pl.Trainer(
            logger=self.comet_logger,
            max_epochs=1, #30
            callbacks=[self.checkpoint_callback, self.early_stopping_callback]
        )
    # Ustawianie Trainer
    def set_trainer(self, trainer : pl.Trainer):
        self.trainer = trainer
    
    # Ustawianie Callbacku
    def set_callback(self, callback : ModelCheckpoint):
        self.checkpoint_callback = callback
    
    # Ustawianie Early Stopping Callback
    def set_early_stopping(self, early_stopping_callback : EarlyStopping):
        self.early_stopping_callback = early_stopping_callback

    # Przejście modelu w tryb ewaluacji
    def evaluate(self):
        return self.model.eval()
    
    # Trenowanie modelu
    def train(self, train_dataloader, valid_dataloader):
        return self.trainer.fit(self.model, train_dataloader, valid_dataloader)
    
    # Testowanie modelu
    def test(self, test_dataloader):
        return self.trainer.test(self.model, test_dataloader)
    
    # Ekstrakcja cech z pojedynczego obrazu
    def single_features_extract(self, image, n_components_reduced: int = 32):
        model = self.model.to(self.device)

        # Load saved PCA model
        pca_path = os.path.join(self.repo_root, 'models', 'pca_model.joblib')
        pca = load(pca_path)

        # Load saved scaler model
        scaler_path = os.path.join(self.repo_root, 'models', 'scaler_model.joblib')
        scaler = load(scaler_path)

        # Przetwarzanie danych
        with torch.no_grad():
            # Jeśli obraz ma wymiary (C, H, W), dodaj wymiar batch
            if len(image.shape) == 3:
                image = image.unsqueeze(0)  # Dodanie wymiaru batch

            # Przenieś obraz na odpowiednie urządzenie
            image = image.to(self.device)

            # Wyciąganie cech z modelu
            extracted_features = model.extract_features(image).cpu().numpy()

            # Spłaszczenie wymiarów przestrzennych
            extracted_features = extracted_features.reshape(extracted_features.shape[0], -1)

            # Redukcja wymiarowości za pomocą PCA
            reduced_features = pca.transform(extracted_features)

            # Normalizacja za pomocą scalera
            normalized_features = scaler.transform(reduced_features)

        # Zwróć znormalizowane cechy (macierz zamiast listy)
        return normalized_features
    
    # Wyciąganie cech z modelu z redukcją wymiarowości
    def multi_features_extract(self, dataloader: DataLoader, n_components_reduced: int = 32, file_name: str = "extracted_features", pca_model_path=None, scaler_model_path=None):
        model = self.model.to(self.device)
        # Konfiguracja PCA
        pca = None

        # Sprawdzamy, czy ścieżka do modelu PCA została podana
        if pca_model_path is not None:
            try:
                pca = load(pca_model_path)  # Załadowanie istniejącego modelu PCA
                print(f"Załadowano model PCA z: {pca_model_path}")
            except Exception as e:
                print(f"Nie udało się załadować modelu PCA z {pca_model_path}: {e}")
                print("Będzie wykonywane dopasowanie PCA na nowych danych.")
                pca = PCA(n_components=n_components_reduced)  # Jeśli nie udało się załadować, tworzymy nowy model PCA
        else:
            pca = PCA(n_components=n_components_reduced)  # Jeśli nie podano ścieżki, tworzymy nowy model PCA

        # Konfiguracja MinMaxScaler 
        scaler = None
        if scaler_model_path is not None:
            try:
                scaler = load(scaler_model_path)  # Załadowanie istniejącego modelu scaler
                print(f"Załadowano model scaler z: {scaler_model_path}")
            except Exception as e:
                print(f"Nie udało się załadować modelu scaler z {scaler_model_path}: {e}")
                print("Będzie wykonywana normalizacja z nowym modelem scaler.")
                scaler = MinMaxScaler(feature_range=(-1, 1))
        else:
            scaler = MinMaxScaler(feature_range=(-1, 1))

        # Listy do przechowywania danych
        features_list = []  # Lista cech
        indices_list = []   # Lista indeksów
        # Flaga wskazująca, czy PCA zostało dopasowane
        is_pca_fitted = False

        with torch.no_grad():
            for _, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
                images, _, indices = batch  # Pobieranie obrazów, etykiet i indeksów
                images = images.to(self.device)

                # Wyciąganie cech
                extracted_features = model.extract_features(images).cpu().numpy()

                # Spłaszczenie wymiarów przestrzennych
                extracted_features = extracted_features.reshape(extracted_features.shape[0], -1)

                # Dopasowanie PCA na pierwszej partii danych i zapis dopasowanego modelu PCA
                if not is_pca_fitted:
                    print("PCA zostanie dopasowane")
                    if pca_model_path is None:  # Jeśli model PCA nie został załadowany, dopasowujemy go na nowych danych
                        pca.fit(extracted_features)
                        dump(pca, os.path.join(self.repo_root, 'models', 'pca_model.joblib'))  # Zapis lokalny modelu PCA
                        print(f"Model PCA zapisany w: {os.path.join(self.repo_root, 'models', 'pca_model.joblib')}")
                        
                    # Wyświetlanie wyjaśnionej wariancji
                    print(f"\n Explained variance (wartości bezwzględne): \n {pca.explained_variance_}")
                    print(f"\n Explained variance ratio (procentowo): \n {pca.explained_variance_ratio_}")
                    # Logowanie wyjaśnionej wariancji do Comet.ml
                    self.log_explained_variance_to_comet(pca.explained_variance_ratio_, n_components_reduced)
                    is_pca_fitted = True

                # Redukcja wymiarowości
                reduced_features = pca.transform(extracted_features)
                # Dodawanie cech i indeksów do list
                features_list.extend(reduced_features)
                indices_list.extend(indices.cpu().numpy())  # Zamiast etykiet, zapisujemy indeksy

        # Normalizacja cech
        features_array = np.array(features_list)
        
        # Dopasowanie scalera tylko na danych treningowych
        if scaler_model_path is None:
            print("Scaler zostanie dopasowany i zapisany.")
            scaler.fit(features_array)
            dump(scaler, os.path.join(self.repo_root, 'models', 'scaler_model.joblib'))  # Zapis lokalny modelu scaler
            print(f"Model scaler zapisany w: {os.path.join(self.repo_root, 'models', 'scaler_model.joblib')}")

        # Normalizacja danych do zakresu [-1, 1]
        normalized_features = scaler.transform(features_array)

        # Tworzenie nazwy pliku na podstawie argumentu
        save_file_name = os.path.join(self.repo_root, 'result', f'{file_name}_features_and_indices.npz')
        os.makedirs(os.path.dirname(save_file_name), exist_ok=True)

        # Zapis w jednym pliku .npz
        np.savez_compressed(save_file_name, features=normalized_features, indices=np.array(indices_list))
        print(f"\n Cechy i indeksy zapisane w: {save_file_name}")

        return save_file_name

    # Ładowanie modelu z checkpoint
    def load_from_checkpoint(self, model_path=None):
        if model_path is None:
            try:
                # Pobierz ścieżkę najlepszego modelu z callbacku checkpoint
                model_path = self.checkpoint_callback.best_model_path
            except Exception as e:
                print(f"Brak dostępnych modeli, podaj ścieżkę dostępu ręcznie! {e}")
                return

        # Sprawdzenie rozszerzenia pliku
        if model_path.endswith(".ckpt"):
            try:
                # Załadowanie modelu z checkpointu .ckpt
                self.model = UNetLightning.load_from_checkpoint(model_path)
                print(f"Model załadowany z checkpointu .ckpt: {model_path}")
            except Exception as e:
                print(f"Nie udało się załadować modelu z .ckpt: {e}")
                return
        elif model_path.endswith(".pth"):
            try:
                # Ręczne ładowanie wag z pliku .pth
                self.model = UNetLightning()  # Tworzenie instancji modelu
                state_dict = torch.load(model_path, map_location=self.device)

                # Usunięcie prefiksu "model.", jeśli występuje w kluczach state_dict
                if any(key.startswith('model.') for key in state_dict.keys()):
                    state_dict = {key.replace('model.', ''): value for key, value in state_dict.items()}

                # Załaduj wagi do modelu
                self.model.model.load_state_dict(state_dict)
                print(f"Model załadowany z pliku .pth: {model_path}")
            except Exception as e:
                print(f"Nie udało się załadować modelu z .pth: {e}")
                return
        else:
            print(f"Nieobsługiwany format pliku: {model_path}")
            return

        # Inicjalizacja obiektu trenera
        self.trainer = pl.Trainer(
            logger=self.comet_logger,
            max_epochs=1,
            callbacks=[self.checkpoint_callback, self.early_stopping_callback]
        )
        print(f"Pomyślnie załadowano model. Trainer ustawiony na {self.trainer.max_epochs} epok.")


    def test_pca_components(self, dataloader: DataLoader, n_components_list: list = [16, 32, 64, 128]):

        model = self.model.to(self.device)

        # Flaga wskazująca, czy PCA zostało dopasowane
        for idx, n_components_reduced in enumerate(tqdm(n_components_list, desc="Processing batches")):
            
            # Konfiguracja PCA
            pca = PCA(n_components=n_components_reduced)

            with torch.no_grad():
                # Wybieramy tylko jeden batch (partię) z dataloadera
                images, _, _ = next(iter(dataloader))  # Pobieranie jednej partii
                images = images.to(self.device)

                # Wyciąganie cech
                extracted_features = model.extract_features(images).cpu().numpy()

                # Spłaszczenie wymiarów przestrzennych
                extracted_features = extracted_features.reshape(extracted_features.shape[0], -1)

                # Dopasowanie PCA
                pca.fit(extracted_features)
                # Logowanie wyjaśnionej wariancji do Comet.ml
                self.log_explained_variance_to_comet(pca.explained_variance_ratio_, n_components_reduced)


    def log_explained_variance_to_comet(self, explained_variance_ratio, n_components):
        # Tworzenie figury z dwoma wykresami (jeden obok drugiego)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 10))  # 1 rząd, 2 kolumny

        # Pierwszy wykres - wyjaśniona wariancja (bar chart)
        components = np.arange(1, len(explained_variance_ratio) + 1)
        ax1.bar(components, explained_variance_ratio * 100, alpha=0.7, color='skyblue', edgecolor='white')
        ax1.set_xlabel('Komponenty PCA', fontsize=14)
        ax1.set_ylabel('Wyjaśniona wariancja (%)', fontsize=14)
        ax1.set_title(f'Wyjaśniona wariancja dla {n_components} komponentów PCA', fontsize=16)
        ax1.grid(axis='y', linestyle='--', alpha=0.7)

        # Drugi wykres - skumulowana wyjaśniona wariancja (bar chart)
        cumulative_variance = np.cumsum(explained_variance_ratio) * 100
        ax2.bar(components, cumulative_variance, alpha=0.7, color='green', edgecolor='white')
        ax2.set_xlabel('Komponenty PCA', fontsize=14)
        ax2.set_ylabel('Skumulowana wyjaśniona wariancja (%)', fontsize=14)
        ax2.set_title(f'Skumulowana wyjaśniona wariancja dla {n_components} komponentów PCA', fontsize=16)
        ax2.grid(axis='y', linestyle='--', alpha=0.7)

        # Dodawanie wartości procentowych na słupkach dla wykresu skumulowanej wariancji
        for i, value in enumerate(cumulative_variance):
            # Dodajemy podpisy co piąty słupek od ostatniego, włącznie z ostatnim
            if (i + 1) % 5 == 0 or i == len(cumulative_variance) - 1:  # Co 5-ty komponent, w tym ostatni
                ax2.text(i + 1, value + 1, f'{value:.1f}%', ha='center', fontsize=11)

        # Logowanie wykresu na Comet
        self.comet_logger.experiment.log_figure(plt.gcf())

        plt.close()

    def summary(self):
        return self.model.summary()