import dash
from dash import Dash, dcc, html, Input, Output

import h5py
import numpy as np
import pandas as pd
from PIL import Image
import os
import sys
import base64
import io

import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision.transforms import v2

repo_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(repo_root, 'extractor'))
sys.path.append(os.path.join(repo_root, 'clustering'))

from feature_extractor import FeatureExtractor
from unet import UNetLightning
from clusterizer import Clusterizer


class WikiArtDataset(Dataset):
    def __init__(self, h5_path: str, mask_h5_path: str, csv_path: str, set_type: str, group_id=None, label_col='style', transform=None):
        self.h5_path = h5_path
        self.mask_h5_path = mask_h5_path
        
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df['set_type'] == set_type]

        if group_id is not None:
            self.df = self.df[self.df['cluster_label'] == group_id]
        
        self.label_col = label_col
        self.transform = transform
        
        self.length = len(self.df)
  
        with h5py.File(self.mask_h5_path, 'r') as mask_h5f:
            self.num_masks = mask_h5f['mask'].shape[0]

    def __len__(self):
        return self.length

    def _open_hdf5(self):
        if not hasattr(self, '_hf') or self._hf is None:
            self._hf = h5py.File(self.h5_path, 'r')

        if not hasattr(self, '_mask_hf') or self._mask_hf is None:
            self._mask_hf = h5py.File(self.mask_h5_path, 'r')

    def _get_random_mask(self):
        mask_idx = np.random.randint(0, self.num_masks)
        mask = self._mask_hf['mask'][mask_idx]
        return mask
    
    def __getitem__(self, idx):
        self._open_hdf5()

        row = self.df.iloc[idx]
        image_idx = row['h5_index']

        label = row[self.label_col]

        image = self._hf['image'][image_idx]
        image = torch.from_numpy(image)

        mask = self._get_random_mask()
        mask = torch.from_numpy(mask).float()
        
        if self.transform:
            image = self.transform(image)

        return image, mask, label

class UNetInpainting(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, use_dropout=False):
        super().__init__()
        self.use_dropout = use_dropout

        self.encoder1 = self.conv_block(in_channels, 16)
        self.encoder2 = self.conv_block(16, 32, pool=True)
        self.encoder3 = self.conv_block(32, 64, pool=True)
        self.encoder4 = self.conv_block(64, 128, pool=True)

        self.bottleneck = self.conv_block(128, 256, pool=True)

        self.decoder4 = self.upconv_block(256, 128)
        self.decoder3 = self.upconv_block(128, 64)
        self.decoder2 = self.upconv_block(64, 32)
        self.decoder1 = self.upconv_block(32, 16)

        self.final_conv = nn.Conv2d(16, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels, pool=False):
        layers = []
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2))
        layers.extend([
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        ])
        if self.use_dropout:
            layers.append(nn.Dropout(0.5))
        return nn.Sequential(*layers)

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x, mask):
        x_with_mask = x * (1 - mask)

        x_with_mask_and_mask = torch.cat([x_with_mask, mask], dim=1)

        e1 = self.encoder1(x_with_mask_and_mask)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        b = self.bottleneck(e4)

        d4 = self.decoder4(b) + e4
        d3 = self.decoder3(d4) + e3
        d2 = self.decoder2(d3) + e2
        d1 = self.decoder1(d2) + e1

        output = self.final_conv(d1)

        output = output * mask + x * (1 - mask)
    
        return output

def to_base64(tensor, is_mask=False):
    arr = tensor.clone().detach().cpu().numpy()
    if not is_mask:
        arr = arr.transpose(1, 2, 0).clip(0, 255).astype(np.uint8)
    else:
        arr = (arr * 255).clip(0, 255).astype(np.uint8)
    pil_img = Image.fromarray(arr.squeeze())
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("utf-8")

def load_model(weights_path, model_class, device='cpu'):
    model = model_class()
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model

model_paths = {
    0: os.path.join(repo_root, 'models', 'inpainting_weights', '0', 'best_weights_epoch_13_val_loss_1.9445-195.pth'),
    1: os.path.join(repo_root, 'models', 'inpainting_weights', '1', 'best_weights_epoch_0_val_loss_3.9911.pth'),
    2: os.path.join(repo_root, 'models', 'inpainting_weights', '2', 'best_weights_epoch_7_val_loss_5.2359-79.pth'),
    3: os.path.join(repo_root, 'models', 'inpainting_weights', '3', 'best_weights_epoch_0_val_loss_4.9286.pth'),
    4: os.path.join(repo_root, 'models', 'inpainting_weights', '4', 'best_weights_epoch_0_val_loss_2.7693-13.pth'),
    5: os.path.join(repo_root, 'models', 'inpainting_weights', '5', 'best_weights_epoch_15_val_loss_2.8780-239.pth')
}

transforms_inpainting = v2.Compose([
    v2.ToDtype(torch.float32, scale=False),
    # v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

transforms_cluster = v2.Compose([
    v2.ToDtype(torch.float32, scale=True),
    # v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

h5_path = os.path.join(repo_root, 'dataset', 'images', 'dataset.h5')
annotations_path = os.path.join(repo_root, 'dataset', 'images', 'annotations.csv')
mask_h5_path = os.path.join(repo_root, 'dataset', 'masks', 'square.h5')

app = Dash(__name__)

# Global index to keep the same image when just randomizing mask
current_image_idx = None

app.layout = html.Div([
    html.Div([
        html.H1('WikiArt Inpainting', style={'textAlign': 'center', 'margin-bottom': '20px'}),
        html.Div([
            html.Button("Losuj obraz", id='random-button', n_clicks=0,
                        style={'margin-right': '10px', 'padding': '10px', 'background-color': '#4CAF50', 'color': 'white', 'border': 'none'}),
            html.Button("Losuj maskę", id='random-mask-button', n_clicks=0,
                        style={'padding': '10px', 'background-color': '#f0ad4e', 'color': 'white', 'border': 'none'})
        ], style={'margin-top': '15px', 'textAlign': 'center'}),
        html.Div(id='cluster-label', style={'margin-top': '15px', 'font-weight': 'bold', 'textAlign': 'center'})
    ], style={'background-color': '#f9f9f9', 'padding': '20px', 'border-bottom': '2px solid #ddd'}),

    html.Div([
        html.Div([
            html.H3("Oryginał", style={'textAlign': 'center'}),
            html.Img(id='original-image', style={'width': '100%', 'border': '1px solid #ddd', 'border-radius': '5px', 'padding': '5px'})
        ], style={'width': '22%', 'display': 'inline-block', 'vertical-align': 'top', 'padding': '10px'}),

        html.Div([
            html.H3("Maska", style={'textAlign': 'center'}),
            html.Img(id='mask-image', style={'width': '100%', 'border': '1px solid #ddd', 'border-radius': '5px', 'padding': '5px'})
        ], style={'width': '22%', 'display': 'inline-block', 'vertical-align': 'top', 'padding': '10px'}),

        html.Div([
            html.H3("Obraz z maską", style={'textAlign': 'center'}),
            html.Img(id='masked-image', style={'width': '100%', 'border': '1px solid #ddd', 'border-radius': '5px', 'padding': '5px'})
        ], style={'width': '22%', 'display': 'inline-block', 'vertical-align': 'top', 'padding': '10px'}),

        html.Div([
            html.H3("Wynik", style={'textAlign': 'center'}),
            html.Img(id='output-image', style={'width': '100%', 'border': '1px solid #ddd', 'border-radius': '5px', 'padding': '5px'})
        ], style={'width': '22%', 'display': 'inline-block', 'vertical-align': 'top', 'padding': '10px'})
    ], style={'textAlign': 'center', 'background-color': '#fff', 'padding': '20px'})
], style={'font-family': 'Arial, sans-serif', 'max-width': '1200px', 'margin': '0 auto'})

@app.callback(
    [
        Output('original-image', 'src'),
        Output('masked-image', 'src'),
        Output('mask-image', 'src'),
        Output('output-image', 'src'),
        Output('cluster-label', 'children')
    ],
    [
        Input('random-button', 'n_clicks'),
        Input('random-mask-button', 'n_clicks')
    ]
)
def update_images(n_clicks_image, n_clicks_mask):
    global current_image_idx

    test_dataset = WikiArtDataset(h5_path, mask_h5_path, annotations_path, 'test', transform=transforms_cluster)

    model_extractor = UNetLightning()
    feature_extractor = FeatureExtractor(model_extractor, 'cpu', None, '')

    best_model_path = os.path.join(repo_root, 'models', 'best_model.pth')
    feature_extractor.load_from_checkpoint(best_model_path)

    feature_extractor.evaluate()


    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else ''

    if trigger_id == 'group-id-dropdown' or trigger_id == 'random-button':
        current_image_idx = np.random.randint(0, len(test_dataset))
    elif current_image_idx is None:
        current_image_idx = np.random.randint(0, len(test_dataset))

    image, _, _ = test_dataset[current_image_idx]

    features = feature_extractor.single_features_extract(image)

    clusterizer = Clusterizer(None)
    n_clusters = 6

    cluster_label = clusterizer.clusterize('test', n_clusters, features)

    model_path = model_paths[cluster_label[0]]
    model = load_model(model_path, UNetInpainting, device='cpu')
    test_dataset_inpainting = WikiArtDataset(h5_path, mask_h5_path, annotations_path, 'test', transform=transforms_inpainting)

    image, mask, _ = test_dataset_inpainting[current_image_idx]

    with torch.no_grad():
        masked_image = image * (1 - mask)
        output = model(image.unsqueeze(0), mask.unsqueeze(0).unsqueeze(0))[0]

    original_b64 = to_base64(image)
    masked_b64 = to_base64(masked_image)
    mask_b64 = to_base64(mask, is_mask=True)
    output_b64 = to_base64(output)

    return original_b64, masked_b64, mask_b64, output_b64, f"Przydzielona grupa: {cluster_label[0]}"

if __name__ == '__main__':
    app.run_server(debug=False)