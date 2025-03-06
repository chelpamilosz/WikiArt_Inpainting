# WikiArt Inpainting  

[2025-01-26 14-04-13.webm](https://github.com/user-attachments/assets/9a85c6cb-88ea-4226-82d8-fdcb8df375d9)  

This project implements an image inpainting system for artwork from the WikiArt dataset using separate models trained on different artwork clusters based on extracted visual features.  

## Dataset
The dataset used for training and evaluation is [Artificio/WikiArt](https://huggingface.co/datasets/Artificio/WikiArt), which contains a diverse collection of artworks. The images were preprocessed by resizing them to 224x224 and saving them as NumPy arrays in an HDF5 file.

## Methodology
### Feature extraction and clustering
The backbone of the model is UNet, which was first trained to extract meaningful features from the images. To reduce dimensionality, PCA was applied to the extracted features. The optimal number of clusters was estimated using the elbow method and silhouette score, leading to a selection of 6 clusters. Clusters were visualized using t-SNE and UMAP.  

![cluster_viz](https://github.com/user-attachments/assets/ac5a42a2-a8e3-4241-b0b6-6ac04ac43c7f)

### Inpainting
The inpainting model was trained using same UNet architecture with additional fourth channel on input for damage mask. Initially, training was conducted on the entire dataset. Subsequently, fine-tuning was performed on subsets of data based on clusters to improve the model’s ability to reconstruct missing regions.

## Examples
![example1](https://github.com/user-attachments/assets/ffc3f569-c603-46c8-9450-782a510e2368)

![example2](https://github.com/user-attachments/assets/e539d257-da20-4e66-bd7b-4dbddfb80bab)

![example3](https://github.com/user-attachments/assets/bd378fbc-bac0-41e4-8a99-61b11220278f)
