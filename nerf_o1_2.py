import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


@torch.no_grad()
def test(hn, hf, dataset, chunk_size=10, img_index=0, nb_bins=192, H=400, W=400):
    """
    Args:
        hn: near plane distance
        hf: far plane distance
        dataset: dataset to render
        chunk_size (int, optional): chunk size for memory efficiency. Defaults to 10.
        img_index (int, optional): image index to render. Defaults to 0.
        nb_bins (int, optional): number of bins for density estimation. Defaults to 192.
        H (int, optional): image height. Defaults to 400.
        W (int, optional): image width. Defaults to 400.
        
    Returns:
        None: None
    """
    ray_origins = dataset[img_index * H * W: (img_index + 1) * H * W, :3]
    ray_directions = dataset[img_index * H * W: (img_index + 1) * H * W, 3:6]

    data = []   # list of regenerated pixel values
    for i in range(int(np.ceil(H / chunk_size))):   # iterate over chunks
        # Get chunk of rays
        ray_origins_ = ray_origins[i * W * chunk_size: (i + 1) * W * chunk_size].to(device)
        ray_directions_ = ray_directions[i * W * chunk_size: (i + 1) * W * chunk_size].to(device)        
        regenerated_px_values = render_rays(model, ray_origins_, ray_directions_, hn=hn, hf=hf, nb_bins=nb_bins)
        data.append(regenerated_px_values)
    img = torch.cat(data).data.cpu().numpy().reshape(H, W, 3)

    plt.figure()
    plt.imshow(img)
    plt.savefig(f'novel_views/img_{img_index}.png', bbox_inches='tight')
    plt.close()


class NerfModel(nn.Module):
    def __init__(self, embedding_dim_pos=10, embedding_dim_direction=4, hidden_dim=128):   
        super(NerfModel, self).__init__()
        
        # Adjusted input dimensions for B-spline encoding
        input_dim_pos = (embedding_dim_pos + 1) * 3  # +1 for including x
        input_dim_dir = (embedding_dim_direction + 1) * 3  # +1 for including d

        self.block1 = nn.Sequential(nn.Linear(input_dim_pos, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), )
        # density estimation
        self.block2 = nn.Sequential(nn.Linear(input_dim_pos + hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim + 1), )
        # color estimation
        self.block3 = nn.Sequential(nn.Linear(input_dim_dir + hidden_dim, hidden_dim // 2), nn.ReLU(), )
        self.block4 = nn.Sequential(nn.Linear(hidden_dim // 2, 3), nn.Sigmoid(), )

        self.embedding_dim_pos = embedding_dim_pos
        self.embedding_dim_direction = embedding_dim_direction
        self.relu = nn.ReLU()

    @staticmethod
    def bspline_basis(x, degree, knots):
        # x: tensor of shape [batch_size]
        # degree: int
        # knots: tensor of shape [num_knots]
        n = knots.shape[0] - degree - 1  # Number of basis functions
        N = []
        for i in range(n + degree):
            N.append(((x >= knots[i]) & (x < knots[i+1])).float())
        N = torch.stack(N, dim=1)
        # For last knot, include x == knots[-1]
        N[:, -1][x == knots[-1]] = 1.0
        # Recursively compute higher-degree basis functions
        for p in range(1, degree + 1):
            N_new = []
            for i in range(n + degree - p):
                denom1 = knots[i + p] - knots[i]
                denom2 = knots[i + p + 1] - knots[i + 1]
                term1 = torch.zeros_like(x)
                term2 = torch.zeros_like(x)
                if denom1 > 0:
                    term1 = ((x - knots[i]) / denom1) * N[:, i]
                if denom2 > 0:
                    term2 = ((knots[i + p + 1] - x) / denom2) * N[:, i + 1]
                N_new.append(term1 + term2)
            N = torch.stack(N_new, dim=1)
        # Return N[:, :n], the first n basis functions
        return N

    def bspline_encoding(self, x, L):
        # x: tensor of shape [batch_size, 3]
        # L: number of basis functions per coordinate
        # Returns: tensor of shape [batch_size, (L + 1) * 3]
        degree = 3  # Cubic B-spline
        n = L
        m = n + degree + 1  # number of knots
        # Assuming x in [-1,1], normalize to [0,1]
        x_normalized = (x + 1) / 2
        # Create knot vector with clamped ends
        knots = torch.linspace(0, 1, n - degree + 1, device=x.device)
        # Add degree multiplicity at the ends
        start = torch.zeros(degree, device=x.device)
        end = torch.ones(degree, device=x.device)
        knots = torch.cat([start, knots, end])
        # Now compute basis functions for each coordinate
        bspline_features = []
        for i in range(3):  # For x, y, z
            xi = x_normalized[:, i]
            N = self.bspline_basis(xi, degree, knots)
            # Include xi itself
            bspline_features.append(torch.cat([xi.unsqueeze(1), N], dim=1))
        # Concatenate features
        bspline_encoding = torch.cat(bspline_features, dim=1)
        return bspline_encoding

    def forward(self, o, d):
        emb_x = self.bspline_encoding(o, self.embedding_dim_pos)  # emb_x: [batch_size, (embedding_dim_pos + 1) * 3]
        emb_d = self.bspline_encoding(d, self.embedding_dim_direction)  # emb_d: [batch_size, (embedding_dim_direction + 1) * 3]
        h = self.block1(emb_x)  # h: [batch_size, hidden_dim]
        tmp = self.block2(torch.cat((h, emb_x), dim=1))  # tmp: [batch_size, hidden_dim + 1]
        h, sigma = tmp[:, :-1], self.relu(tmp[:, -1])  # h: [batch_size, hidden_dim], sigma: [batch_size]
        h = self.block3(torch.cat((h, emb_d), dim=1))  # h: [batch_size, hidden_dim // 2]
        c = self.block4(h)  # c: [batch_size, 3]
        return c, sigma


def compute_accumulated_transmittance(alphas):
    accumulated_transmittance = torch.cumprod(alphas, 1)
    return torch.cat((torch.ones((accumulated_transmittance.shape[0], 1), device=alphas.device),
                      accumulated_transmittance[:, :-1]), dim=-1)


def render_rays(nerf_model, ray_origins, ray_directions, hn=0, hf=0.5, nb_bins=192):
    device = ray_origins.device
    
    t = torch.linspace(hn, hf, nb_bins, device=device).expand(ray_origins.shape[0], nb_bins)
    # Perturb sampling along each ray.
    mid = (t[:, :-1] + t[:, 1:]) / 2.
    lower = torch.cat((t[:, :1], mid), -1)
    upper = torch.cat((mid, t[:, -1:]), -1)
    u = torch.rand(t.shape, device=device)
    t = lower + (upper - lower) * u  # [batch_size, nb_bins]
    delta = torch.cat((t[:, 1:] - t[:, :-1], torch.tensor([1e10], device=device).expand(ray_origins.shape[0], 1)), -1)

    # Compute the 3D points along each ray
    x = ray_origins.unsqueeze(1) + t.unsqueeze(2) * ray_directions.unsqueeze(1)   # [batch_size, nb_bins, 3]
    # Expand the ray_directions tensor to match the shape of x
    ray_directions = ray_directions.expand(nb_bins, ray_directions.shape[0], 3).transpose(0, 1) 

    colors, sigma = nerf_model(x.reshape(-1, 3), ray_directions.reshape(-1, 3))
    colors = colors.reshape(x.shape)
    sigma = sigma.reshape(x.shape[:-1])

    alpha = 1 - torch.exp(-sigma * delta)  # [batch_size, nb_bins]
    weights = compute_accumulated_transmittance(1 - alpha).unsqueeze(2) * alpha.unsqueeze(2)
    # Compute the pixel values as a weighted sum of colors along each ray
    c = (weights * colors).sum(dim=1)
    weight_sum = weights.sum(-1).sum(-1)  # Regularization for white background 
    return c + 1 - weight_sum.unsqueeze(-1)


def train(nerf_model, optimizer, scheduler, data_loader, device='cpu', hn=0, hf=1, nb_epochs=int(1e5),
          nb_bins=192, H=400, W=400):
    training_loss = []
    for epoch in tqdm(range(nb_epochs)):
        for batch in data_loader:
            ray_origins = batch[:, :3].to(device)
            ray_directions = batch[:, 3:6].to(device)
            ground_truth_px_values = batch[:, 6:].to(device)
            
            regenerated_px_values = render_rays(nerf_model, ray_origins, ray_directions, hn=hn, hf=hf, nb_bins=nb_bins) 
            loss = ((ground_truth_px_values - regenerated_px_values) ** 2).sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_loss.append(loss.item())
        scheduler.step()

        # Optionally, save images during training
        # for img_index in range(200):
        #     test(hn, hf, testing_dataset, img_index=img_index, nb_bins=nb_bins, H=H, W=W)
    return training_loss


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    training_dataset = torch.from_numpy(np.load('training_data.pkl', allow_pickle=True))
    testing_dataset = torch.from_numpy(np.load('testing_data.pkl', allow_pickle=True))
    model = NerfModel(hidden_dim=256).to(device)
    model_optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(model_optimizer, milestones=[2, 4, 8], gamma=0.5)
    data_loader = DataLoader(training_dataset, batch_size=1024, shuffle=True)
    train(model, model_optimizer, scheduler, data_loader, nb_epochs=16, device=device, hn=2, hf=6, nb_bins=192, H=400,
          W=400)
