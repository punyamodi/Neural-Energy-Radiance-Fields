import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os
from skimage.metrics import structural_similarity as compare_ssim
import lpips  # pip install lpips
import csv

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

##########################################
# B-Spline encoding routines (unchanged)
##########################################
def bspline_basis(x, degree, n_basis, eps=1e-8):
    """
    Compute the B-spline basis functions for a 1D tensor x.
    Assumes x is in [0,1]. Uses the Cox–de Boor recursion.
    """
    if x.dim() > 1:
        x = x.squeeze(-1)
    N = x.shape[0]
    n_knots = n_basis + degree + 1
    if n_basis - degree - 1 > 0:
        interior = torch.linspace(0, 1, n_basis - degree + 1, device=x.device)[1:-1]
    else:
        interior = torch.tensor([], device=x.device)
    knots = torch.cat([torch.zeros(degree+1, device=x.device), interior, torch.ones(degree+1, device=x.device)])
    x_exp = x.unsqueeze(1)
    knots_i = knots[:n_basis].view(1, -1)
    knots_ip1 = knots[1:n_basis+1].view(1, -1)
    B = ((x_exp >= knots_i) & (x_exp < knots_ip1)).float()
    idx = (x == 1)
    if idx.any():
        B[idx] = 0.0
        B[idx, -1] = 1.0
    for d in range(1, degree+1):
        left_num = x_exp - knots[:n_basis].view(1, -1)
        left_denom = knots[d:n_basis+d] - knots[:n_basis]
        left_denom = left_denom.view(1, -1)
        left_term = (left_num / (left_denom + eps)) * B
        right_num = knots[d+1:n_basis+d+1].view(1, -1) - x_exp
        right_denom = knots[d+1:n_basis+d+1] - knots[1:n_basis+1]
        right_denom = right_denom.view(1, -1)
        B_right = (right_num / (right_denom + eps)) * torch.cat([B[:, 1:], torch.zeros(N, 1, device=x.device)], dim=1)
        B = left_term + B_right
    return B

def bspline_encoding(x, degree, n_basis, include_input=True):
    """
    Apply B-spline encoding to each coordinate of x.
    """
    encodings = []
    for i in range(x.shape[1]):
        xi = x[:, i:i+1]
        xi_norm = (xi + 1) / 2  
        basis = bspline_basis(xi_norm.squeeze(-1), degree, n_basis)
        if include_input:
            encodings.append(torch.cat([xi, basis], dim=1))
        else:
            encodings.append(basis)
    return torch.cat(encodings, dim=1)

###############################################
# Kolmogorov-Arnold Network (KA) module
###############################################
class KolmogorovArnoldLayer(nn.Module):
    """
    A simple implementation of a Kolmogorov-Arnold layer.
    It approximates a function f(x) (with x in R^d) as
        f(x) = sum_{q=0}^{2*d} φ_q ( sum_{p=1}^{d} ψ_{q,p}(x_p) )
    where each ψ_{q,p} and φ_q is implemented as a small MLP.
    """
    def __init__(self, input_dim, output_dim, num_terms=None, hidden_dim=32):
        """
        Args:
            input_dim: the dimension of the input (d)
            output_dim: the dimension of the output
            num_terms: number of terms in the sum; by default 2*input_dim+1.
            hidden_dim: hidden dimension for the univariate networks.
        """
        super(KolmogorovArnoldLayer, self).__init__()
        self.input_dim = input_dim
        if num_terms is None:
            num_terms = 2 * input_dim + 1
        self.num_terms = num_terms
        self.hidden_dim = hidden_dim

        # Create a collection of univariate networks ψ_{q,p} for each term q and coordinate p.
        self.psi = nn.ModuleList()
        for q in range(self.num_terms):
            psi_q = nn.ModuleList()
            for p in range(input_dim):
                psi_q.append(nn.Sequential(
                    nn.Linear(1, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1)
                ))
            self.psi.append(psi_q)
        
        # Create the outer functions φ_q.
        self.phi = nn.ModuleList()
        for q in range(self.num_terms):
            self.phi.append(nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            ))
    
    def forward(self, x):
        # x: [N, input_dim]
        N = x.shape[0]
        device = x.device
        out = torch.zeros(N, self.phi[0][-1].out_features, device=device)
        for q in range(self.num_terms):
            sum_val = torch.zeros(N, 1, device=device)
            for p in range(self.input_dim):
                xp = x[:, p:p+1]  # shape: (N,1)
                sum_val = sum_val + self.psi[q][p](xp)
            term = self.phi[q](sum_val)
            out = out + term
        return out

###############################################
# NeRF model with options for B-spline and KA encoding
###############################################
class NerfModel(nn.Module):
    def __init__(self, 
                 embedding_dim_pos=10, 
                 embedding_dim_direction=4, 
                 hidden_dim=128,
                 use_bspline=False,
                 bspline_degree=3,
                 bspline_num_basis=None,   # if None, will be set to 2*embedding_dim_pos
                 bspline_degree_dir=3,
                 bspline_num_basis_dir=None,  # if None, will be set to 2*embedding_dim_direction
                 use_kolmogorov=False,
                 ka_hidden_dim=32  # hidden dimension for KA sub-networks
                ):   
        super(NerfModel, self).__init__()
        self.use_bspline = use_bspline
        self.use_kolmogorov = use_kolmogorov

        # Choose the input dimension for positions.
        if self.use_kolmogorov:
            # With KA we use the raw coordinate (assumed in [-1,1]).
            input_dim_pos = 3
            self.ka_layer = KolmogorovArnoldLayer(input_dim=3, output_dim=hidden_dim, 
                                                   num_terms=2*3+1, hidden_dim=ka_hidden_dim)
            # In this case, we do not use a standard block1 MLP.
            self.block1 = nn.Identity()
        elif self.use_bspline:
            self.bspline_degree = bspline_degree
            self.bspline_num_basis = bspline_num_basis if bspline_num_basis is not None else 2 * embedding_dim_pos
            input_dim_pos = 3 * (1 + self.bspline_num_basis)
            self.block1 = nn.Sequential(
                nn.Linear(input_dim_pos, hidden_dim), 
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), 
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), 
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), 
                nn.ReLU(),
            )
        else:
            input_dim_pos = 3 * (1 + 2 * embedding_dim_pos)
            self.block1 = nn.Sequential(
                nn.Linear(input_dim_pos, hidden_dim), 
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), 
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), 
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), 
                nn.ReLU(),
            )
        
        # For directions we still use either B-spline or sine-cosine.
        if self.use_bspline:
            self.bspline_degree_dir = bspline_degree_dir
            self.bspline_num_basis_dir = bspline_num_basis_dir if bspline_num_basis_dir is not None else 2 * embedding_dim_direction
            input_dim_dir = 3 * (1 + self.bspline_num_basis_dir)
        else:
            input_dim_dir = 3 * (1 + 2 * embedding_dim_direction)
        
        # Block2 processes the combination of the (encoded) positional input and its hidden representation.
        self.block2 = nn.Sequential(
            nn.Linear(input_dim_pos + hidden_dim, hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim + 1),  # last element is sigma
        )
        self.block3 = nn.Sequential(
            nn.Linear(input_dim_dir + hidden_dim, hidden_dim // 2), 
            nn.ReLU(),
        )
        self.block4 = nn.Sequential(
            nn.Linear(hidden_dim // 2, 3), 
            nn.Sigmoid(),  # produce colors in [0,1]
        )
        self.embedding_dim_pos = embedding_dim_pos
        self.embedding_dim_direction = embedding_dim_direction
        self.relu = nn.ReLU()

    @staticmethod
    def positional_encoding(x, L):
        out = [x]
        for j in range(L):
            out.append(torch.sin((2 ** j) * x))
            out.append(torch.cos((2 ** j) * x))
        return torch.cat(out, dim=1)

    def forward(self, o, d):
        # For positions:
        if self.use_kolmogorov:
            # Use raw coordinates and pass through the KA layer.
            emb_x = o  # raw input of shape [N,3]
            hidden = self.ka_layer(emb_x)  # output shape: [N, hidden_dim]
        elif self.use_bspline:
            emb_x = bspline_encoding(o, self.bspline_degree, self.bspline_num_basis, include_input=True)
            hidden = self.block1(emb_x)
        else:
            emb_x = self.positional_encoding(o, self.embedding_dim_pos)
            hidden = self.block1(emb_x)
        # For directions:
        if self.use_bspline:
            emb_d = bspline_encoding(d, self.bspline_degree_dir, self.bspline_num_basis_dir, include_input=True)
        else:
            emb_d = self.positional_encoding(d, self.embedding_dim_direction)
        # Combine positional encoding and hidden representation.
        tmp = self.block2(torch.cat((emb_x, hidden), dim=1))
        h, sigma = tmp[:, :-1], self.relu(tmp[:, -1])
        h = self.block3(torch.cat((h, emb_d), dim=1))
        c = self.block4(h)
        return c, sigma

###############################################
# Rendering and testing routines
###############################################
@torch.no_grad()
def test(hn, hf, dataset, model, lpips_loss, chunk_size=10, img_index=0, nb_bins=192, H=400, W=400, 
         save_path="novel_views", csv_path="test_metrics.csv", epoch=0):
    """
    Render an image from a novel view given an index.
    Also computes PSNR, SSIM, and LPIPS metrics against the ground truth.
    The metrics are logged to a CSV file.
    """
    device = next(model.parameters()).device
    start = img_index * H * W
    end = (img_index + 1) * H * W
    ray_origins = dataset[start:end, :3]
    ray_directions = dataset[start:end, 3:6]
    gt_pixels = dataset[start:end, 6:]
    
    data = []
    for i in range(int(np.ceil(H / chunk_size))):
        s = i * W * chunk_size
        e = (i + 1) * W * chunk_size
        ro_chunk = ray_origins[s:e].to(device)
        rd_chunk = ray_directions[s:e].to(device)
        rendered_chunk = render_rays(model, ro_chunk, rd_chunk, hn=hn, hf=hf, nb_bins=nb_bins)
        data.append(rendered_chunk)
    pred = torch.cat(data)  # [N, 3]
    img = pred.data.cpu().numpy().reshape(H, W, 3)
    
    gt = gt_pixels.data.cpu().numpy().reshape(H, W, 3)
    
    os.makedirs(save_path, exist_ok=True)
    plt.figure()
    plt.imshow(img)
    plt.axis("off")
    plt.savefig(os.path.join(save_path, f'img_{img_index}.png'), bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # Compute PSNR.
    mse = np.mean((img - gt) ** 2)
    psnr = -10 * np.log10(mse + 1e-8)
    # Compute SSIM.
    ssim = compare_ssim(gt, img, channel_axis=-1, data_range=1.0, win_size=7)
    # Compute LPIPS.
    pred_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device) * 2 - 1.
    gt_tensor = torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0).to(device) * 2 - 1.
    lpips_val = lpips_loss(pred_tensor, gt_tensor).item()

    # Log the metrics to a CSV file.
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["epoch", "view_index", "psnr", "ssim", "lpips"])
        writer.writerow([epoch, img_index, psnr, ssim, lpips_val])

    print(f"View {img_index}: PSNR: {psnr:.3f} dB, SSIM: {ssim:.3f}, LPIPS: {lpips_val:.3f}")
    return psnr, ssim, lpips_val

def compute_accumulated_transmittance(alphas):
    accumulated_transmittance = torch.cumprod(alphas, 1)
    return torch.cat((torch.ones((accumulated_transmittance.shape[0], 1), device=alphas.device),
                      accumulated_transmittance[:, :-1]), dim=-1)

def render_rays(nerf_model, ray_origins, ray_directions, hn=0, hf=0.5, nb_bins=192):
    """
    Render rays by sampling along each ray and combining the predicted colors
    weighted by their transmittance.
    """
    device = ray_origins.device
    t = torch.linspace(hn, hf, nb_bins, device=device).expand(ray_origins.shape[0], nb_bins)
    mid = (t[:, :-1] + t[:, 1:]) / 2.
    lower = torch.cat((t[:, :1], mid), -1)
    upper = torch.cat((mid, t[:, -1:]), -1)
    u = torch.rand(t.shape, device=device)
    t = lower + (upper - lower) * u  # [batch_size, nb_bins]
    delta = torch.cat((t[:, 1:] - t[:, :-1],
                       torch.tensor([1e10], device=device).expand(ray_origins.shape[0], 1)), -1)

    x = ray_origins.unsqueeze(1) + t.unsqueeze(2) * ray_directions.unsqueeze(1)
    ray_directions_exp = ray_directions.expand(nb_bins, ray_directions.shape[0], 3).transpose(0, 1) 

    colors, sigma = nerf_model(x.reshape(-1, 3), ray_directions_exp.reshape(-1, 3))
    colors = colors.reshape(x.shape)
    sigma = sigma.reshape(x.shape[:-1])

    alpha = 1 - torch.exp(-sigma * delta)  # [batch_size, nb_bins]
    weights = compute_accumulated_transmittance(1 - alpha).unsqueeze(2) * alpha.unsqueeze(2)
    c = (weights * colors).sum(dim=1)
    weight_sum = weights.sum(-1).sum(-1)
    return c + 1 - weight_sum.unsqueeze(-1)

def train(nerf_model, optimizer, scheduler, data_loader, device='cpu', hn=0, hf=1, nb_epochs=int(1e5),
          nb_bins=192, H=400, W=400, testing_dataset=None, lpips_loss=None, csv_path="test_metrics.csv"):
    training_loss = []
    for epoch in range(nb_epochs):
        pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{nb_epochs}", leave=False)
        for batch in pbar:
            ray_origins = batch[:, :3].to(device)
            ray_directions = batch[:, 3:6].to(device)
            ground_truth_px_values = batch[:, 6:].to(device)
            
            regenerated_px_values = render_rays(nerf_model, ray_origins, ray_directions, hn=hn, hf=hf, nb_bins=nb_bins) 
            loss = ((ground_truth_px_values - regenerated_px_values) ** 2).sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_loss.append(loss.item())
            pbar.set_postfix(loss=loss.item())
        scheduler.step()

        if testing_dataset is not None:
            num_views = testing_dataset.shape[0] // (H * W)
            metrics = []
            for img_index in range(num_views):
                psnr, ssim_val, lpips_val = test(hn, hf, testing_dataset, nerf_model, lpips_loss,
                                                 img_index=img_index, nb_bins=nb_bins, H=H, W=W, 
                                                 csv_path=csv_path, epoch=epoch+1)
                metrics.append((psnr, ssim_val, lpips_val))
            avg_psnr = np.mean([m[0] for m in metrics])
            avg_ssim = np.mean([m[1] for m in metrics])
            avg_lpips = np.mean([m[2] for m in metrics])
            print(f"After epoch {epoch+1}: Avg PSNR: {avg_psnr:.3f} dB, Avg SSIM: {avg_ssim:.3f}, Avg LPIPS: {avg_lpips:.3f}")
        print(f"Epoch {epoch+1}/{nb_epochs} completed.")
    return training_loss

###############################################
# Main – choose whether to use B-spline or KA encoding
###############################################
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    training_dataset = torch.from_numpy(np.load('training_data.pkl', allow_pickle=True))
    testing_dataset = torch.from_numpy(np.load('testing_data.pkl', allow_pickle=True))
    
    # Choose your encoding:
    # To use B-spline encoding, set use_bspline=True.
    # To use Kolmogorov-Arnold encoding, set use_kolmogorov=True.
    # (These options are mutually exclusive.)
    model = NerfModel(hidden_dim=256, 
                      embedding_dim_pos=10, 
                      embedding_dim_direction=4,
                      use_bspline=True,       # set to True to use B-spline encoding
                      use_kolmogorov=True,       # set to True to use KA encoding
                      bspline_degree=3,       
                      bspline_num_basis=20,   
                      bspline_degree_dir=3,
                      bspline_num_basis_dir=8,
                      ka_hidden_dim=32          # hidden dimension for KA networks
                     ).to(device)
    
    model_optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(model_optimizer, milestones=[2, 4, 8], gamma=0.5)
    data_loader = DataLoader(training_dataset, batch_size=1024, shuffle=True)
    
    lpips_loss = lpips.LPIPS(net='alex').to(device)
    
    # Run training (here for 1 epoch; adjust nb_epochs as needed)
    train(model, model_optimizer, scheduler, data_loader, nb_epochs=1, device=device, 
          hn=2, hf=6, nb_bins=192, H=400, W=400, testing_dataset=testing_dataset, 
          lpips_loss=lpips_loss, csv_path="test_metrics.csv")
