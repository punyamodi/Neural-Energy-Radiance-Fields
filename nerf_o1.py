import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Optimized test function with adjustments for image dimensions and chunk processing
@torch.no_grad()
def test(hn, hf, dataset, chunk_size=10, img_index=0, nb_bins=32, H=400, W=400):
    """
    Renders an image using the trained NeRF model.

    Args:
        hn (float): Near plane distance.
        hf (float): Far plane distance.
        dataset (torch.Tensor): Dataset containing ray origins and directions.
        chunk_size (int, optional): Number of rows processed per chunk for memory efficiency. Defaults to 10.
        img_index (int, optional): Index of the image to render. Defaults to 0.
        nb_bins (int, optional): Number of samples along each ray. Defaults to 32.
        H (int, optional): Image height. Defaults to 400.
        W (int, optional): Image width. Defaults to 400.

    Returns:
        None
    """
    device = next(model.parameters()).device
    start_idx = img_index * H * W
    end_idx = (img_index + 1) * H * W
    ray_origins = dataset[start_idx:end_idx, :3].to(device)
    ray_directions = dataset[start_idx:end_idx, 3:6].to(device)

    data = []
    total_pixels = H * W
    chunk_pixels = chunk_size * W
    num_chunks = (total_pixels + chunk_pixels - 1) // chunk_pixels

    for i in range(num_chunks):
        start = i * chunk_pixels
        end = min(start + chunk_pixels, total_pixels)
        ray_origins_chunk = ray_origins[start:end]
        ray_directions_chunk = ray_directions[start:end]
        regenerated_px_values = render_rays(
            model, ray_origins_chunk, ray_directions_chunk, hn=hn, hf=hf, nb_bins=nb_bins
        )
        data.append(regenerated_px_values)
    img = torch.cat(data).data.cpu().numpy().reshape(H, W, 3)

    plt.figure()
    plt.imshow(img)
    plt.savefig(f'novel_views/img_{img_index}.png', bbox_inches='tight')
    plt.close()

# Optimized NeRF model with reduced complexity
class NerfModel(nn.Module):
    def __init__(self, embedding_dim_pos=6, embedding_dim_direction=2, hidden_dim=128):
        """
        Neural Radiance Field (NeRF) model.

        Args:
            embedding_dim_pos (int, optional): Positional encoding dimension for positions. Defaults to 6.
            embedding_dim_direction (int, optional): Positional encoding dimension for directions. Defaults to 2.
            hidden_dim (int, optional): Dimension of hidden layers. Defaults to 128.
        """
        super(NerfModel, self).__init__()

        # Reduced embedding dimensions and hidden dimensions for faster computation
        self.block1 = nn.Sequential(
            nn.Linear(embedding_dim_pos * 6 + 3, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.block2 = nn.Sequential(
            nn.Linear(embedding_dim_pos * 6 + hidden_dim + 3, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim + 1),
        )
        self.block3 = nn.Sequential(
            nn.Linear(embedding_dim_direction * 6 + hidden_dim + 3, hidden_dim // 2), nn.ReLU(),
        )
        self.block4 = nn.Sequential(
            nn.Linear(hidden_dim // 2, 3), nn.Sigmoid(),
        )

        self.embedding_dim_pos = embedding_dim_pos
        self.embedding_dim_direction = embedding_dim_direction
        self.relu = nn.ReLU()

    @staticmethod
    def positional_encoding(x, L):
        """
        Applies positional encoding to the input.

        Args:
            x (torch.Tensor): Input tensor.
            L (int): Number of frequency bands.

        Returns:
            torch.Tensor: Positional encoded tensor.
        """
        out = [x]
        for j in range(L):
            out.append(torch.sin(2 ** j * x))
            out.append(torch.cos(2 ** j * x))
        return torch.cat(out, dim=1)

    def forward(self, o, d):
        """
        Forward pass of the NeRF model.

        Args:
            o (torch.Tensor): Sampled positions.
            d (torch.Tensor): Ray directions.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Colors and densities.
        """
        emb_x = self.positional_encoding(o, self.embedding_dim_pos)
        emb_d = self.positional_encoding(d, self.embedding_dim_direction)
        h = self.block1(emb_x)
        tmp = self.block2(torch.cat((h, emb_x), dim=1))
        h, sigma = tmp[:, :-1], self.relu(tmp[:, -1])
        h = self.block3(torch.cat((h, emb_d), dim=1))
        c = self.block4(h)
        return c, sigma

def compute_accumulated_transmittance(alphas):
    """
    Computes the accumulated transmittance along the ray.

    Args:
        alphas (torch.Tensor): Alpha values for each sample along the ray.

    Returns:
        torch.Tensor: Accumulated transmittance.
    """
    accumulated_transmittance = torch.cumprod(alphas, 1)
    return torch.cat((
        torch.ones((accumulated_transmittance.shape[0], 1), device=alphas.device),
        accumulated_transmittance[:, :-1]
    ), dim=-1)

def render_rays(nerf_model, ray_origins, ray_directions, hn=0, hf=0.5, nb_bins=32):
    """
    Renders rays using the NeRF model.

    Args:
        nerf_model (NerfModel): Trained NeRF model.
        ray_origins (torch.Tensor): Origins of the rays.
        ray_directions (torch.Tensor): Directions of the rays.
        hn (float, optional): Near plane distance. Defaults to 0.
        hf (float, optional): Far plane distance. Defaults to 0.5.
        nb_bins (int, optional): Number of samples along each ray. Defaults to 32.

    Returns:
        torch.Tensor: Rendered pixel values.
    """
    device = ray_origins.device

    t = torch.linspace(hn, hf, nb_bins, device=device).expand(ray_origins.shape[0], nb_bins)
    # Perturb sampling along each ray
    mid = (t[:, :-1] + t[:, 1:]) / 2.
    lower = torch.cat((t[:, :1], mid), -1)
    upper = torch.cat((mid, t[:, -1:]), -1)
    u = torch.rand(t.shape, device=device)
    t = lower + (upper - lower) * u  # [batch_size, nb_bins]
    delta = torch.cat((
        t[:, 1:] - t[:, :-1],
        torch.tensor([1e10], device=device).expand(ray_origins.shape[0], 1)
    ), -1)

    # Compute the 3D points along each ray
    x = ray_origins.unsqueeze(1) + t.unsqueeze(2) * ray_directions.unsqueeze(1)  # [batch_size, nb_bins, 3]
    ray_directions_expanded = ray_directions.unsqueeze(1).expand(-1, nb_bins, -1).reshape(-1, 3)

    colors, sigma = nerf_model(x.reshape(-1, 3), ray_directions_expanded)
    colors = colors.reshape(x.shape)
    sigma = sigma.reshape(x.shape[:-1])

    alpha = 1 - torch.exp(-sigma * delta)
    weights = compute_accumulated_transmittance(1 - alpha) * alpha
    c = (weights.unsqueeze(-1) * colors).sum(dim=1)
    weight_sum = weights.sum(-1)
    return c + 1 - weight_sum.unsqueeze(-1)

def train(nerf_model, optimizer, scheduler, data_loader, device='cpu', hn=0, hf=1, nb_epochs=1,
          nb_bins=32, H=400, W=400):
    """
    Trains the NeRF model.

    Args:
        nerf_model (NerfModel): NeRF model to train.
        optimizer (torch.optim.Optimizer): Optimizer.
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
        data_loader (DataLoader): DataLoader for training data.
        device (str, optional): Device to use ('cpu' or 'cuda'). Defaults to 'cpu'.
        hn (float, optional): Near plane distance. Defaults to 0.
        hf (float, optional): Far plane distance. Defaults to 1.
        nb_epochs (int, optional): Number of epochs to train. Defaults to 1.
        nb_bins (int, optional): Number of samples along each ray. Defaults to 32.
        H (int, optional): Image height. Defaults to 400.
        W (int, optional): Image width. Defaults to 400.

    Returns:
        List[float]: Training loss history.
    """
    training_loss = []
    for epoch in tqdm(range(nb_epochs), desc='Training'):
        epoch_loss = 0.0
        for batch in data_loader:
            ray_origins = batch[:, :3].to(device)
            ray_directions = batch[:, 3:6].to(device)
            ground_truth_px_values = batch[:, 6:].to(device)

            regenerated_px_values = render_rays(
                nerf_model, ray_origins, ray_directions, hn=hn, hf=hf, nb_bins=nb_bins
            )
            loss = ((ground_truth_px_values - regenerated_px_values) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        avg_epoch_loss = epoch_loss / len(data_loader)
        training_loss.append(avg_epoch_loss)
        print(f'Epoch {epoch + 1}/{nb_epochs}, Loss: {avg_epoch_loss:.6f}')
    return training_loss

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load datasets
    training_dataset = torch.from_numpy(np.load('training_data.pkl', allow_pickle=True))
    testing_dataset = torch.from_numpy(np.load('testing_data.pkl', allow_pickle=True))

    print(f"Training dataset size: {training_dataset.shape}")
    print(f"Testing dataset size: {testing_dataset.shape}")
    
    # Ensure datasets have enough samples for images of size H x W
    # Each image requires H * W samples (400 * 400 = 160,000 samples)
    # Adjusted dataset sizes to match image dimensions
    training_dataset = training_dataset[:160000]  # Reduced training dataset size for faster training
    #testing_dataset = testing_dataset[:160000]    # Use first 160,000 samples for one test image

    # Create model with reduced complexity
    model = NerfModel(
        embedding_dim_pos=5,        # Reduced from 10 to 6 for faster computation
        embedding_dim_direction=4,  # Reduced from 4 to 2
        hidden_dim=128             # Reduced from 256 to 128
    ).to(device)

    # Optimizer and scheduler with adjusted parameters
    model_optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        model_optimizer, milestones=[2, 4, 8], gamma=0.5
    )

    # DataLoader with adjusted batch size for memory efficiency
    data_loader = DataLoader(
        training_dataset, batch_size=1024, shuffle=True
    )

    # Training with reduced epochs and number of bins for faster execution
    training_loss = train(
        model, model_optimizer, scheduler, data_loader,
        nb_epochs=300,               # Reduced number of epochs from int(1e5) to 1
        device=device, hn=2, hf=6,
        nb_bins=32,                # Reduced from 192 to 32
        H=400, W=400
    )

    # Testing after training
    test(
        hn=2, hf=6, dataset=testing_dataset,
        img_index=0, nb_bins=32,   # Consistent nb_bins with training
        H=400, W=400
    )
