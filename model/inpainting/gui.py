import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import numpy as np
import torch
import torch.nn as nn
import h5py
from torch.utils.data import DataLoader, Dataset


class WikiArtDataset(Dataset):
    def __init__(self, h5_path: str, transform=None, mask_transform=None):
        """
        A dataset for WikiArt data integrated with HDF5 image and mask reading.

        Parameters:
            h5_path (str): Path to the HDF5 file containing images and masks.
            transform (callable, optional): A function to transform the images.
            mask_transform (callable, optional): A function to transform the masks.
        """
        self.h5_path = h5_path
        self.transform = transform
        self.mask_transform = mask_transform

        with h5py.File(self.h5_path, 'r') as h5f:
            self.length = len(h5f['images'])

    def __len__(self):
        """
        Returns the total number of elements in the dataset.

        Returns:
            int: The number of images in the dataset.
        """
        return self.length

    def _open_hdf5(self):
        """
        Opens the HDF5 file if it hasn't been opened yet.
        """
        if not hasattr(self, '_hf') or self._hf is None:
            self._hf = h5py.File(self.h5_path, 'r')

    def __getitem__(self, idx):
        """
        Retrieves an image and its corresponding mask based on the given index.

        Parameters:
            idx (int): The index of the element to retrieve.

        Returns:
            tuple: A tuple containing the image (torch.Tensor) and the mask (torch.Tensor).
        """
        self._open_hdf5()

        image = self._hf['images'][idx]
        mask = self._hf['masks'][idx]

        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).float()

        if self.transform:
            image = self.transform(image)

        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask

class UNetInpainting(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, use_dropout=False):
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
        masked_image = x * (1 - mask)

        e1 = self.encoder1(masked_image)  # No downsampling here
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        b = self.bottleneck(e4)

        d4 = self.decoder4(b) + e4
        d3 = self.decoder3(d4) + e3
        d2 = self.decoder2(d3) + e2
        d1 = self.decoder1(d2) + e1

        output = self.final_conv(d1)

        return output

def load_model(checkpoint_path, model_class, device='cpu'):
    model = model_class()
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model

app = dash.Dash(__name__)

checkpoint_path = r'C:\Users\Milosz\vscodeProjects\WikiArt_Inpainting\model\inpainting\best_weights_epoch_98_val_loss_37.2326-6533.pth'
test_dataset_path = r'C:\Users\Milosz\vscodeProjects\WikiArt_Inpainting\model\inpainting\test.h5'

test_dataset = WikiArtDataset(test_dataset_path)
model = load_model(checkpoint_path, UNetInpainting, device='cpu')

def process_sample(dataset, model, idx, device='cpu'):
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    model.to(device)

    images, masks = dataset[idx]
    images, masks = images.to(device).unsqueeze(0), masks.to(device).unsqueeze(0)

    masked_image = images * (1 - masks.unsqueeze(1))

    with torch.no_grad():
        output = model(images, masks.unsqueeze(1))

    return images[0], masked_image[0], masks[0], output[0]

# Layout aplikacji
app.layout = html.Div([
    html.H1('WikiArt Inpainting GUI'),
    html.Label('Wybierz indeks obrazu:'),
    dcc.Dropdown(
        id='image-index-dropdown',
        options=[{'label': f'Image {i}', 'value': i} for i in range(len(test_dataset))],
        value=0
    ),
    html.Div([
        dcc.Graph(id='original-image', config={'displayModeBar': False}),
        dcc.Graph(id='masked-image', config={'displayModeBar': False}),
        dcc.Graph(id='mask-image', config={'displayModeBar': False}),
        dcc.Graph(id='output-image', config={'displayModeBar': False})
    ], style={'display': 'flex', 'flex-wrap': 'wrap', 'justify-content': 'space-around'})
])

# Callback do aktualizacji wykresów
@app.callback(
    [Output('original-image', 'figure'),
     Output('masked-image', 'figure'),
     Output('mask-image', 'figure'),
     Output('output-image', 'figure')],
    [Input('image-index-dropdown', 'value')]
)

def update_visualization(selected_index):
    original, masked, mask, output = process_sample(test_dataset, model, selected_index, device='cpu')

    original_fig = px.imshow(original.permute(1, 2, 0).cpu().numpy().astype(np.uint8), title='Original')
    masked_fig = px.imshow(masked.permute(1, 2, 0).cpu().numpy().astype(np.uint8), title='Masked')
    mask_fig = px.imshow(mask.squeeze(0).cpu().numpy().astype(np.uint8), title='Mask', color_continuous_scale='gray')
    output_fig = px.imshow(output.permute(1, 2, 0).detach().cpu().numpy().clip(0, 255).astype(np.uint8), title='Output')

    for fig in [original_fig, masked_fig, mask_fig, output_fig]:
        fig.update_layout(
            coloraxis_showscale=False,
            margin=dict(l=0, r=0, t=30, b=0),
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            autosize=True
        )

    return original_fig, masked_fig, mask_fig, output_fig

if __name__ == '__main__':
    app.run_server(debug=True)
