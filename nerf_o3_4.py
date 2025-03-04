import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Optional: import evaluation libraries
import lpips  # pip install lpips
from skimage.metrics import structural_similarity as compare_ssim
import math


class NerfModel(nn.Module):
    def __init__(self, embedding_dim_pos=10, embedding_dim_direction=4, hidden_dim=128,
                 use_bspline=False, bspline_num_knots_pos=16, bspline_num_knots_dir=8,
                 domain_min=-1, domain_max=1):
        """
        Args:
            embedding_dim_pos: number of frequencies for positional encoding (if not using B-Spline)
            embedding_dim_direction: same for direction
            hidden_dim: hidden layer dimension
            use_bspline: if True, use B-Spline encoding instead of sinusoidal encoding.
            bspline_num_knots_pos: number of knots for position encoding via B-Spline.
            bspline_num_knots_dir: number of knots for direction encoding via B-Spline.
            domain_min, domain_max: domain for B-Spline normalization.
        """
        super(NerfModel, self).__init__()
        self.use_bspline = use_bspline
        self.domain_min = domain_min
        self.domain_max = domain_max

        if self.use_bspline:
            pos_enc_dim = 3 * bspline_num_knots_pos
            dir_enc_dim = 3 * bspline_num_knots_dir
            self.bspline_num_knots_pos = bspline_num_knots_pos
            self.bspline_num_knots_dir = bspline_num_knots_dir
        else:
            pos_enc_dim = embedding_dim_pos * 6 + 3  # original encoding: x, sin, cos for each frequency per dim
            dir_enc_dim = embedding_dim_direction * 6 + 3

        # First block processes the encoded position
        self.block1 = nn.Sequential(
            nn.Linear(pos_enc_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        # Second block for density estimation, concatenating encoded position with hidden features
        self.block2 = nn.Sequential(
            nn.Linear(pos_enc_dim + hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim + 1)
        )
        # Third block for color estimation, concatenating hidden features with encoded view directions
        self.block3 = nn.Sequential(
            nn.Linear(hidden_dim + dir_enc_dim, hidden_dim // 2), nn.ReLU()
        )
        self.block4 = nn.Sequential(
            nn.Linear(hidden_dim // 2, 3), nn.Sigmoid()
        )

    @staticmethod
    def positional_encoding(x, L):
        # x: [batch, dims]
        out = [x]
        for j in range(L):
            out.append(torch.sin(2 ** j * x))
            out.append(torch.cos(2 ** j * x))
        return torch.cat(out, dim=1)

    @staticmethod
    def cubic_bspline(u):
        abs_u = torch.abs(u)
        return torch.where(
            abs_u < 1,
            2/3 - abs_u**2 + 0.5 * abs_u**3,
            torch.where(
                abs_u < 2,
                ((2 - abs_u)**3) / 6,
                torch.zeros_like(u)
            )
        )

    def bspline_encoding(self, x, num_knots):
        """
        Computes a B-Spline encoding for each coordinate.
        Assumes x is of shape [batch, dims]. The coordinates are normalized from domain_min to domain_max.
        Returns: tensor of shape [batch, dims * num_knots]
        """
        # Normalize x to [0, 1]
        x_norm = (x - self.domain_min) / (self.domain_max - self.domain_min)
        # Create equally spaced knots in [0, 1]
        knots = torch.linspace(0, 1, num_knots, device=x.device)
        h = 1.0 / (num_knots - 1)
        # Compute basis values for each coordinate
        # x_norm: [batch, dims] -> [batch, dims, 1]
        x_exp = x_norm.unsqueeze(-1)
        # knots: [num_knots] -> [1, 1, num_knots]
        knots_exp = knots.view(1, 1, -1)
        u = (x_exp - knots_exp) / h  # [batch, dims, num_knots]
        basis = NerfModel.cubic_bspline(u)
        # Flatten last two dimensions: result shape [batch, dims * num_knots]
        return basis.view(x.shape[0], -1)

    def forward(self, o, d):
        # o: [batch, 3] ray origins or 3D points
        # d: [batch, 3] ray directions
        if self.use_bspline:
            emb_x = self.bspline_encoding(o, self.bspline_num_knots_pos)
            emb_d = self.bspline_encoding(d, self.bspline_num_knots_dir)
        else:
            emb_x = self.positional_encoding(o, L=((self.block1[0].in_features - 3) // 6))
            emb_d = self.positional_encoding(d, L=((self.block3[0].in_features - self.block1[-1].out_features) // 6))
        h = self.block1(emb_x)
        tmp = self.block2(torch.cat((emb_x, h), dim=1))
        h, sigma = tmp[:, :-1], torch.relu(tmp[:, -1])
        h = self.block3(torch.cat((h, emb_d), dim=1))
        c = self.block4(h)
        return c, sigma


def compute_accumulated_transmittance(alphas):
    # alphas: [batch, nb_bins]
    accumulated_transmittance = torch.cumprod(alphas, dim=1)
    # Prepend ones so that T(0)=1 for each ray
    ones = torch.ones((alphas.shape[0], 1), device=alphas.device)
    return torch.cat((ones, accumulated_transmittance[:, :-1]), dim=-1)


def render_rays(nerf_model, ray_origins, ray_directions, hn=0, hf=0.5, nb_bins=192):
    device = ray_origins.device
    t = torch.linspace(hn, hf, nb_bins, device=device).expand(ray_origins.shape[0], nb_bins)
    # Perturb sampling along each ray
    mid = (t[:, :-1] + t[:, 1:]) / 2.
    lower = torch.cat((t[:, :1], mid), dim=-1)
    upper = torch.cat((mid, t[:, -1:]), dim=-1)
    u = torch.rand(t.shape, device=device)
    t = lower + (upper - lower) * u  # [batch, nb_bins]
    delta = torch.cat((t[:, 1:] - t[:, :-1],
                       torch.tensor([1e10], device=device).expand(ray_origins.shape[0], 1)), dim=-1)

    # Compute 3D sample points along each ray
    x = ray_origins.unsqueeze(1) + t.unsqueeze(2) * ray_directions.unsqueeze(1)  # [batch, nb_bins, 3]
    # For the viewing directions, expand per sample point
    ray_directions_exp = ray_directions.unsqueeze(1).expand_as(x)  # [batch, nb_bins, 3]

    colors, sigma = nerf_model(x.reshape(-1, 3), ray_directions_exp.reshape(-1, 3))
    colors = colors.reshape(x.shape)
    sigma = sigma.reshape(x.shape[:-1])
    alpha = 1 - torch.exp(-sigma * delta)  # [batch, nb_bins]
    weights = compute_accumulated_transmittance(1 - alpha).unsqueeze(2) * alpha.unsqueeze(2)
    # Weighted sum along the ray gives the pixel color; add background correction
    c = (weights * colors).sum(dim=1)
    weight_sum = weights.sum(dim=(1, 2))  # [batch]
    return c + 1 - weight_sum.unsqueeze(-1)


@torch.no_grad()
def render_image(nerf_model, dataset, img_index, hn, hf, nb_bins, H, W, chunk_size=10):
    """
    Render an image given the test dataset.
    """
    rays = dataset[img_index * H * W: (img_index + 1) * H * W]
    ray_origins = rays[:, :3]
    ray_directions = rays[:, 3:6]
    out_chunks = []
    for i in range(int(np.ceil(H / chunk_size))):
        start = i * W * chunk_size
        end = min((i + 1) * W * chunk_size, H * W)
        ro_chunk = ray_origins[start:end].to(device)
        rd_chunk = ray_directions[start:end].to(device)
        out_chunks.append(render_rays(model, ro_chunk, rd_chunk, hn=hn, hf=hf, nb_bins=nb_bins))
    img = torch.cat(out_chunks).cpu().numpy().reshape(H, W, 3)
    return img


def train(nerf_model, optimizer, scheduler, data_loader, device='cpu',
          hn=0, hf=1, nb_epochs=1000, nb_bins=192, H=400, W=400):
    training_loss = []
    nerf_model.train()
    for epoch in range(nb_epochs):
        epoch_losses = []
        # Use tqdm to show progress for batches in the current epoch
        with tqdm(data_loader, desc=f"Epoch {epoch+1}/{nb_epochs}", leave=False) as pbar:
            for batch in pbar:
                ray_origins = batch[:, :3].to(device)
                ray_directions = batch[:, 3:6].to(device)
                ground_truth_px = batch[:, 6:].to(device)
                regenerated_px = render_rays(nerf_model, ray_origins, ray_directions, hn=hn, hf=hf, nb_bins=nb_bins)
                loss = torch.mean((ground_truth_px - regenerated_px) ** 2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())
                # Update progress bar with loss information
                pbar.set_postfix(loss=loss.item())
                training_loss.append(loss.item())
        scheduler.step()
        avg_loss = np.mean(epoch_losses)
        epoch_pct = (epoch + 1) / nb_epochs * 100
        print(f"Epoch {epoch+1}/{nb_epochs} completed ({epoch_pct:.1f}% overall). Average Loss: {avg_loss:.6f}")
    return training_loss


def evaluate(nerf_model, testing_dataset, device, hn, hf, nb_bins, H, W):
    """
    Evaluate the model on the testing dataset and compute average PSNR, SSIM, and LPIPS.
    """
    nerf_model.eval()
    lpips_fn = lpips.LPIPS(net='alex').to(device)
    psnr_list, ssim_list, lpips_list = [], [], []
    num_images = testing_dataset.shape[0] // (H * W)
    for img_index in range(num_images):
        rays = testing_dataset[img_index * H * W: (img_index + 1) * H * W]
        # Ground truth pixel colors (assumed to be stored from column 6 onward)
        gt = rays[:, 6:]
        gt_img = gt.cpu().numpy().reshape(H, W, 3)
        # Render the image (using chunked rendering for memory efficiency)
        rendered_img = render_image(nerf_model, testing_dataset, img_index, hn, hf, nb_bins, H, W)
        
        # Compute PSNR
        mse = np.mean((gt_img - rendered_img) ** 2)
        psnr_val = 10 * math.log10(1.0 / mse) if mse > 0 else 100
        psnr_list.append(psnr_val)
        
        # Compute SSIM (multichannel)
        ssim_val = compare_ssim(gt_img, rendered_img, multichannel=True, data_range=gt_img.max() - gt_img.min())
        ssim_list.append(ssim_val)
        
        # Compute LPIPS; convert images to tensors with shape [1, 3, H, W] in range [-1, 1]
        rendered_tensor = torch.from_numpy(rendered_img).permute(2, 0, 1).unsqueeze(0).to(device) * 2 - 1
        gt_tensor = torch.from_numpy(gt_img).permute(2, 0, 1).unsqueeze(0).to(device) * 2 - 1
        with torch.no_grad():
            lpips_val = lpips_fn(gt_tensor, rendered_tensor).item()
        lpips_list.append(lpips_val)

    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    avg_lpips = np.mean(lpips_list)
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"Average LPIPS: {avg_lpips:.4f}")


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load your datasets (ensure your .pkl files are correctly formatted)
    training_dataset = torch.from_numpy(np.load('training_data.pkl', allow_pickle=True))
    testing_dataset = torch.from_numpy(np.load('testing_data.pkl', allow_pickle=True))
    
    # Initialize the model; set use_bspline=True to activate the new encoding.
    model = NerfModel(hidden_dim=256, use_bspline=True,
                      bspline_num_knots_pos=16, bspline_num_knots_dir=8,
                      domain_min=-1, domain_max=1).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2, 4, 8], gamma=0.5)
    data_loader = DataLoader(training_dataset, batch_size=1024, shuffle=True)
    
    # Train the model (adjust nb_epochs as needed)
    train(model, optimizer, scheduler, data_loader, device=device, hn=2, hf=6, nb_epochs=1, nb_bins=192, H=400, W=400)
    
    # Evaluate after training
    evaluate(model, testing_dataset, device=device, hn=2, hf=6, nb_bins=192, H=400, W=400)
