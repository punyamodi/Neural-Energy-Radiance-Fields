import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

#############################
# B-Spline encoding routines
#############################

def bspline_basis(x, degree, n_basis, eps=1e-8):
    """
    Compute the B-spline basis functions for a 1D tensor x.
    Assumes x is in [0,1]. Uses the Cox–de Boor recursion.

    Args:
        x: tensor of shape (N,) containing evaluation points in [0,1]
        degree: the degree of the B-spline (e.g. 3 for cubic)
        n_basis: number of basis functions.
        eps: small constant to avoid division by zero.

    Returns:
        B: tensor of shape (N, n_basis) with the B-spline basis values.
    """
    # Ensure x is a 1D tensor of shape (N,)
    if x.dim() > 1:
        x = x.squeeze(-1)
    N = x.shape[0]
    n_knots = n_basis + degree + 1
    # Build an open uniform knot vector in [0,1]
    if n_basis - degree - 1 > 0:
        # interior knots: choose (n_basis - degree - 1) knots uniformly in (0,1)
        interior = torch.linspace(0, 1, n_basis - degree + 1, device=x.device)[1:-1]
    else:
        interior = torch.tensor([], device=x.device)
    knots = torch.cat([torch.zeros(degree+1, device=x.device), interior, torch.ones(degree+1, device=x.device)])
    # Compute degree-0 basis functions.
    x_exp = x.unsqueeze(1)  # shape (N, 1)
    knots_i = knots[:n_basis].view(1, -1)
    knots_ip1 = knots[1:n_basis+1].view(1, -1)
    B = ((x_exp >= knots_i) & (x_exp < knots_ip1)).float()  # shape (N, n_basis)
    # Special case: for x==1, assign the last basis function to 1.
    idx = (x == 1)
    if idx.any():
        B[idx] = 0.0
        B[idx, -1] = 1.0

    # Recursively compute basis functions for degrees 1,2,...,degree.
    for d in range(1, degree+1):
        # left term: (x - knots[i]) / (knots[i+d] - knots[i]) * B_{i,d-1}
        left_num = x_exp - knots[:n_basis].view(1, -1)
        left_denom = knots[d:n_basis+d] - knots[:n_basis]
        left_denom = left_denom.view(1, -1)
        left_term = (left_num / (left_denom + eps)) * B

        # right term: (knots[i+d+1] - x) / (knots[i+d+1] - knots[i+1]) * B_{i+1,d-1}
        right_num = knots[d+1:n_basis+d+1].view(1, -1) - x_exp
        right_denom = knots[d+1:n_basis+d+1] - knots[1:n_basis+1]
        right_denom = right_denom.view(1, -1)
        # B_{i+1,d-1}: pad with zeros so that dimensions match.
        B_right = (right_num / (right_denom + eps)) * torch.cat([B[:, 1:], torch.zeros(N, 1, device=x.device)], dim=1)
        # Fix: Replace the undefined 'right_term' with 'B_right'
        B = left_term + B_right
    return B

def bspline_encoding(x, degree, n_basis, include_input=True):
    """
    Apply B-spline encoding to each coordinate of x.
    Assumes that the input x (of shape [N, D]) is in the range [-1,1]. It is first
    normalized to [0,1] (since B-spline basis functions are defined on [0,1]).
    
    For each coordinate, the raw value is (optionally) concatenated with the evaluated
    B-spline basis (of dimension n_basis), so that each coordinate yields (1+n_basis) features.
    
    Args:
        x: tensor of shape (N, D)
        degree: degree of the B-spline (e.g. 3 for cubic)
        n_basis: number of basis functions per coordinate.
        include_input: if True, the raw coordinate is concatenated with the basis values.
        
    Returns:
        Tensor of shape (N, D*(1+n_basis)) if include_input is True, else (N, D*n_basis)
    """
    encodings = []
    for i in range(x.shape[1]):
        xi = x[:, i:i+1]
        # Normalize from [-1,1] to [0,1].
        xi_norm = (xi + 1) / 2  
        basis = bspline_basis(xi_norm.squeeze(-1), degree, n_basis)
        if include_input:
            encodings.append(torch.cat([xi, basis], dim=1))
        else:
            encodings.append(basis)
    return torch.cat(encodings, dim=1)

###########################################
# NeRF model with option for B-spline encoding
###########################################

class NerfModel(nn.Module):
    def __init__(self, 
                 embedding_dim_pos=10, 
                 embedding_dim_direction=4, 
                 hidden_dim=128,
                 use_bspline=False,
                 bspline_degree=3,
                 bspline_num_basis=None,   # if None, will be set to 2*embedding_dim_pos
                 bspline_degree_dir=3,
                 bspline_num_basis_dir=None  # if None, will be set to 2*embedding_dim_direction
                ):   
        super(NerfModel, self).__init__()
        
        self.use_bspline = use_bspline
        if self.use_bspline:
            self.bspline_num_basis = bspline_num_basis if bspline_num_basis is not None else 2 * embedding_dim_pos
            self.bspline_degree = bspline_degree
            self.bspline_num_basis_dir = bspline_num_basis_dir if bspline_num_basis_dir is not None else 2 * embedding_dim_direction
            self.bspline_degree_dir = bspline_degree_dir
            # Each coordinate becomes (1+n_basis) features.
            input_dim_pos = 3 * (1 + self.bspline_num_basis)
            input_dim_dir = 3 * (1 + self.bspline_num_basis_dir)
        else:
            # Standard sine–cosine positional encoding.
            input_dim_pos = 3 * (1 + 2 * embedding_dim_pos)
            input_dim_dir = 3 * (1 + 2 * embedding_dim_direction)
        
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
        if self.use_bspline:
            # Use B-spline encoding (assuming inputs in [-1,1]).
            emb_x = bspline_encoding(o, self.bspline_degree, self.bspline_num_basis, include_input=True)
            emb_d = bspline_encoding(d, self.bspline_degree_dir, self.bspline_num_basis_dir, include_input=True)
        else:
            emb_x = self.positional_encoding(o, self.embedding_dim_pos)
            emb_d = self.positional_encoding(d, self.embedding_dim_direction)
        h = self.block1(emb_x)
        tmp = self.block2(torch.cat((h, emb_x), dim=1))
        h, sigma = tmp[:, :-1], self.relu(tmp[:, -1])
        h = self.block3(torch.cat((h, emb_d), dim=1))
        c = self.block4(h)
        return c, sigma

############################################
# Rendering and training routines
############################################

@torch.no_grad()
def test(hn, hf, dataset, chunk_size=10, img_index=0, nb_bins=192, H=400, W=400):
    """
    Render an image from a novel view.
    """
    ray_origins = dataset[img_index * H * W: (img_index + 1) * H * W, :3]
    ray_directions = dataset[img_index * H * W: (img_index + 1) * H * W, 3:6]

    data = []
    for i in range(int(np.ceil(H / chunk_size))):
        ray_origins_ = ray_origins[i * W * chunk_size: (i + 1) * W * chunk_size].to(device)
        ray_directions_ = ray_directions[i * W * chunk_size: (i + 1) * W * chunk_size].to(device)        
        regenerated_px_values = render_rays(model, ray_origins_, ray_directions_, hn=hn, hf=hf, nb_bins=nb_bins)
        data.append(regenerated_px_values)
    img = torch.cat(data).data.cpu().numpy().reshape(H, W, 3)

    plt.figure()
    plt.imshow(img)
    plt.savefig(f'novel_views/img_{img_index}.png', bbox_inches='tight')
    plt.close()

def compute_accumulated_transmittance(alphas):
    accumulated_transmittance = torch.cumprod(alphas, 1)
    return torch.cat((torch.ones((accumulated_transmittance.shape[0], 1), device=alphas.device),
                      accumulated_transmittance[:, :-1]), dim=-1)

def render_rays(nerf_model, ray_origins, ray_directions, hn=0, hf=0.5, nb_bins=192):
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
    # Expand ray_directions to match the shape of x.
    ray_directions_exp = ray_directions.expand(nb_bins, ray_directions.shape[0], 3).transpose(0, 1) 

    colors, sigma = nerf_model(x.reshape(-1, 3), ray_directions_exp.reshape(-1, 3))
    colors = colors.reshape(x.shape)
    sigma = sigma.reshape(x.shape[:-1])

    alpha = 1 - torch.exp(-sigma * delta)  # [batch_size, nb_bins]
    weights = compute_accumulated_transmittance(1 - alpha).unsqueeze(2) * alpha.unsqueeze(2)
    # Compute the pixel values as a weighted sum of colors along each ray.
    c = (weights * colors).sum(dim=1)
    weight_sum = weights.sum(-1).sum(-1)  # Regularization for white background 
    return c + 1 - weight_sum.unsqueeze(-1)

def train(nerf_model, optimizer, scheduler, data_loader, device='cpu', hn=0, hf=1, nb_epochs=int(1e5),
          nb_bins=192, H=400, W=400, testing_dataset=None):
    training_loss = []
    for epoch in range(nb_epochs):
        # Create a tqdm progress bar for the inner loop over batches.
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

        # Optionally, render test images every epoch.
        if testing_dataset is not None:
            for img_index in range(200):
                test(hn, hf, testing_dataset, img_index=img_index, nb_bins=nb_bins, H=H, W=W)
        print(f"Epoch {epoch+1}/{nb_epochs} completed.")
    return training_loss

####################################
# Main – choose whether to use B-spline encoding or not
####################################
if __name__ == '__main__':
    device = 'cuda'
    
    training_dataset = torch.from_numpy(np.load('training_data.pkl', allow_pickle=True))
    testing_dataset = torch.from_numpy(np.load('testing_data.pkl', allow_pickle=True))
    
    # Set use_bspline=True to use the B-spline encoding.
    model = NerfModel(hidden_dim=256, 
                      embedding_dim_pos=10, 
                      embedding_dim_direction=4,
                      use_bspline=True,       # Use B-spline encoding
                      bspline_degree=3,       # Cubic B-spline
                      bspline_num_basis=20,   # e.g. 2*embedding_dim_pos (10*2)
                      bspline_degree_dir=3,
                      bspline_num_basis_dir=8 # e.g. 2*embedding_dim_direction (4*2)
                     ).to(device)
    
    model_optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(model_optimizer, milestones=[2, 4, 8], gamma=0.5)
    data_loader = DataLoader(training_dataset, batch_size=1024, shuffle=True)
    # Run training for 1 epoch (adjust nb_epochs as needed)
    train(model, model_optimizer, scheduler, data_loader, nb_epochs=1, device=device, 
          hn=2, hf=6, nb_bins=192, H=400, W=400, testing_dataset=testing_dataset)
