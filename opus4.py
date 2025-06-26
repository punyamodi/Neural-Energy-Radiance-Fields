import os
import torch
import numpy as np
from tqdm import tqdm # Ensure tqdm is imported
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# ::: START OMP ERROR FIX :::
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# ::: END OMP ERROR FIX :::

# Global variable for device, needs to be set in __main__
device = 'cpu'


class BsplineEncoding(nn.Module):
    def __init__(self, input_dims, num_bases_per_dim, degree=3, min_val=-1.0, max_val=1.0, include_input=True):
        super().__init__()
        self.input_dims = input_dims
        self.num_bases_per_dim = num_bases_per_dim
        self.degree = degree
        
        if self.degree != 3:
            raise ValueError("Only cubic B-splines (degree 3) are currently implemented.")
        if self.num_bases_per_dim < self.degree + 1: # Need at least 4 bases for cubic
             raise ValueError(
                f"num_bases_per_dim ({self.num_bases_per_dim}) must be at least degree + 1 ({self.degree + 1})."
            )

        self.min_val = float(min_val)
        self.max_val = float(max_val)
        self.include_input = include_input
        
        self.scale_factor = (self.num_bases_per_dim - self.degree) / (self.max_val - self.min_val)

    def get_output_dim(self):
        if self.include_input:
            return self.input_dims * (1 + self.num_bases_per_dim)
        else:
            return self.input_dims * self.num_bases_per_dim

    def forward(self, x): # x: (batch, input_dims)
        batch_size = x.shape[0]
        x_device = x.device

        x_scaled = (x - self.min_val) * self.scale_factor
        epsilon = 1e-6
        x_scaled = torch.clamp(x_scaled, 0, self.num_bases_per_dim - self.degree - epsilon)

        idx_int = torch.floor(x_scaled).long()  # (batch, input_dims)
        u = x_scaled - idx_int.float()          # (batch, input_dims)

        u_sq = u * u
        u_cu = u_sq * u
        
        c0 = (1 - u)**3 / 6.0
        c1 = (3*u_cu - 6*u_sq + 4) / 6.0
        c2 = (-3*u_cu + 3*u_sq + 3*u + 1) / 6.0
        c3 = u_cu / 6.0
        
        coeffs_active = torch.stack([c0, c1, c2, c3], dim=-1)
        
        output_features_list = []

        if self.include_input:
            for d_in in range(self.input_dims):
                output_features_list.append(x[:, d_in].unsqueeze(1))
                feat_d = torch.zeros(batch_size, self.num_bases_per_dim, device=x_device)
                idx_scatter = torch.stack([
                    idx_int[:, d_in],
                    idx_int[:, d_in] + 1,
                    idx_int[:, d_in] + 2,
                    idx_int[:, d_in] + 3
                ], dim=-1)
                feat_d.scatter_(dim=1, index=idx_scatter, src=coeffs_active[:, d_in, :])
                output_features_list.append(feat_d)
        else:
            for d_in in range(self.input_dims):
                feat_d = torch.zeros(batch_size, self.num_bases_per_dim, device=x_device)
                idx_scatter = torch.stack([
                    idx_int[:, d_in],
                    idx_int[:, d_in] + 1,
                    idx_int[:, d_in] + 2,
                    idx_int[:, d_in] + 3
                ], dim=-1)
                feat_d.scatter_(dim=1, index=idx_scatter, src=coeffs_active[:, d_in, :])
                output_features_list.append(feat_d)

        return torch.cat(output_features_list, dim=1)


@torch.no_grad()
def test(hn, hf, dataset, model_to_test, chunk_size=10, img_index=0, nb_bins=192, H=400, W=400):
    # Calculate start and end indices for rays
    start_idx = img_index * H * W
    end_idx = (img_index + 1) * H * W

    # Check if requested img_index is valid for the dataset
    if start_idx >= len(dataset):
        print(f"Warning: img_index {img_index} is out of bounds for the testing dataset (max index {len(dataset)//(H*W) -1}). Skipping render.")
        return

    ray_origins = dataset[start_idx:end_idx, :3]
    ray_directions = dataset[start_idx:end_idx, 3:6]

    data = []
    effective_H = ray_origins.shape[0] // W # Actual number of rows for this image index, might be less than H if chunked data
    
    # Iterate over chunks for memory efficiency
    # The number of chunks should be based on the actual number of ray origins fetched for this image.
    num_rays_for_image = ray_origins.shape[0]
    
    for i in range(int(np.ceil(num_rays_for_image / (W * chunk_size)))):
        chunk_start = i * W * chunk_size
        chunk_end = min((i + 1) * W * chunk_size, num_rays_for_image) # Ensure not to go beyond available rays
        
        if chunk_start >= chunk_end: # Should not happen if logic is correct but good for safety
            continue

        ray_origins_ = ray_origins[chunk_start:chunk_end].to(device)
        ray_directions_ = ray_directions[chunk_start:chunk_end].to(device)
        
        if ray_origins_.shape[0] == 0: # Skip if chunk is empty
            continue
            
        regenerated_px_values = render_rays(model_to_test, ray_origins_, ray_directions_, hn=hn, hf=hf, nb_bins=nb_bins)
        data.append(regenerated_px_values)
    
    if not data: # If no data was processed (e.g. empty ray set)
        print(f"Warning: No data processed for img_index {img_index}. Skipping image generation.")
        return

    img_tensor = torch.cat(data)
    
    # Expected number of pixels H * W. If img_tensor is smaller, it means something went wrong or not enough rays.
    # For safety, reshape to the actual number of pixels rendered.
    # The number of pixels rendered should be num_rays_for_image
    try:
        img = img_tensor.data.cpu().numpy().reshape(-1, W, 3) # Reshape based on actual pixels and width
    except RuntimeError as e:
        print(f"Error reshaping image for img_index {img_index}: {e}")
        print(f"  Expected shape around ({H}, {W}, 3), got tensor of size {img_tensor.shape}")
        print(f"  Number of rays for image: {num_rays_for_image}, W: {W}")
        return


    plt.figure()
    plt.imshow(img)
    os.makedirs('novel_views', exist_ok=True)
    plt.savefig(f'novel_views/img_{img_index}.png', bbox_inches='tight')
    plt.close()


class NerfModel(nn.Module):
    def __init__(self, embedding_dim_pos=10, embedding_dim_direction=4, hidden_dim=128, use_bspline=False,
                 bspline_pos_min_max=(-2.0, 2.0), bspline_dir_min_max=(-1.0, 1.0)):
        super(NerfModel, self).__init__()
        
        self.use_bspline = use_bspline
        self.embedding_dim_pos = embedding_dim_pos
        self.embedding_dim_direction = embedding_dim_direction

        if use_bspline:
            num_bases_pos = embedding_dim_pos * 2
            num_bases_dir = embedding_dim_direction * 2
            min_b_spline_bases = 3 + 1
            if num_bases_pos < min_b_spline_bases:
                if embedding_dim_pos * 2 < min_b_spline_bases:
                   self.embedding_dim_pos = (min_b_spline_bases + 1) // 2
                   num_bases_pos = self.embedding_dim_pos * 2

            if num_bases_dir < min_b_spline_bases:
                if embedding_dim_direction * 2 < min_b_spline_bases:
                   self.embedding_dim_direction = (min_b_spline_bases + 1) // 2
                   num_bases_dir = self.embedding_dim_direction * 2
            
            self.bspline_pos_encoder = BsplineEncoding(
                input_dims=3, num_bases_per_dim=num_bases_pos, degree=3, 
                min_val=bspline_pos_min_max[0], max_val=bspline_pos_min_max[1], include_input=True
            )
            self.bspline_dir_encoder = BsplineEncoding(
                input_dims=3, num_bases_per_dim=num_bases_dir, degree=3,
                min_val=bspline_dir_min_max[0], max_val=bspline_dir_min_max[1], include_input=True
            )
            encoded_pos_dim = self.bspline_pos_encoder.get_output_dim()
            encoded_dir_dim = self.bspline_dir_encoder.get_output_dim()
        else:
            encoded_pos_dim = 3 + self.embedding_dim_pos * 6
            encoded_dir_dim = 3 + self.embedding_dim_direction * 6

        self.block1 = nn.Sequential(nn.Linear(encoded_pos_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), )
        self.block2 = nn.Sequential(nn.Linear(hidden_dim + encoded_pos_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim + 1), )
        self.block3 = nn.Sequential(nn.Linear(hidden_dim + encoded_dir_dim, hidden_dim // 2), nn.ReLU(), )
        self.block4 = nn.Sequential(nn.Linear(hidden_dim // 2, 3), nn.Sigmoid(), )
        self.relu = nn.ReLU()

    @staticmethod
    def positional_encoding(x, L):
        out = [x]
        for j in range(L):
            out.append(torch.sin(2 ** j * x))
            out.append(torch.cos(2 ** j * x))
        return torch.cat(out, dim=1)

    def forward(self, o, d):
        if self.use_bspline:
            emb_x = self.bspline_pos_encoder(o)
            emb_d = self.bspline_dir_encoder(d)
        else:
            emb_x = self.positional_encoding(o, self.embedding_dim_pos)
            emb_d = self.positional_encoding(d, self.embedding_dim_direction)
        
        h = self.block1(emb_x) 
        tmp = self.block2(torch.cat((h, emb_x), dim=1)) 
        h_feat, sigma = tmp[:, :-1], self.relu(tmp[:, -1])
        h = self.block3(torch.cat((h_feat, emb_d), dim=1)) 
        c = self.block4(h) 
        return c, sigma


def compute_accumulated_transmittance(alphas):
    accumulated_transmittance = torch.cumprod(alphas, 1)
    return torch.cat((torch.ones((accumulated_transmittance.shape[0], 1), device=alphas.device),
                      accumulated_transmittance[:, :-1]), dim=-1)


def render_rays(nerf_model, ray_origins, ray_directions, hn=0, hf=0.5, nb_bins=192):
    ray_device = ray_origins.device 
    t = torch.linspace(hn, hf, nb_bins, device=ray_device).expand(ray_origins.shape[0], nb_bins)
    mid = (t[:, :-1] + t[:, 1:]) / 2.
    lower = torch.cat((t[:, :1], mid), -1)
    upper = torch.cat((mid, t[:, -1:]), -1)
    u = torch.rand(t.shape, device=ray_device)
    t = lower + (upper - lower) * u
    delta = torch.cat((t[:, 1:] - t[:, :-1], torch.tensor([1e10], device=ray_device).expand(ray_origins.shape[0], 1)), -1)

    x = ray_origins.unsqueeze(1) + t.unsqueeze(2) * ray_directions.unsqueeze(1)
    ray_directions_expanded = ray_directions.unsqueeze(1).expand(-1, nb_bins, -1)

    colors, sigma = nerf_model(x.reshape(-1, 3), ray_directions_expanded.reshape(-1, 3))
    colors = colors.reshape(x.shape)
    sigma = sigma.reshape(x.shape[:-1])

    alpha = 1 - torch.exp(-sigma * delta)
    weights = compute_accumulated_transmittance(1 - alpha).unsqueeze(2) * alpha.unsqueeze(2)
    c = (weights * colors).sum(dim=1)
    weight_sum = weights.sum(-1).sum(-1)
    return c + 1 - weight_sum.unsqueeze(-1)


# MODIFIED TRAIN FUNCTION
def train(nerf_model, optimizer, scheduler, data_loader, testing_dataset_for_test_fn, 
          num_test_renders_per_epoch=200, # New parameter to control number of test images
          device_train='cpu', hn=0, hf=1, nb_epochs=int(1e5),
          nb_bins=192, H=400, W=400):
    training_loss = []
    
    # Calculate how many images are actually available in the test set
    # This assumes all test images have HxW rays.
    if H * W == 0:
        max_available_test_images = 0 # Avoid division by zero if H or W is 0
    else:
        max_available_test_images = len(testing_dataset_for_test_fn) // (H * W)
    
    num_images_to_render_epoch = min(num_test_renders_per_epoch, max_available_test_images)
    if num_test_renders_per_epoch > max_available_test_images:
        print(f"Warning: Requested {num_test_renders_per_epoch} test renders, but only {max_available_test_images} are available in the dataset. Rendering {max_available_test_images}.")


    for epoch_idx in tqdm(range(nb_epochs), desc="Overall Epochs"):
        epoch_loss_sum = 0.0
        num_batches = 0
        nerf_model.train() # Set model to training mode
        
        batch_pbar = tqdm(data_loader, desc=f"Epoch {epoch_idx + 1}/{nb_epochs}", leave=False, unit="batch")
        
        for batch in batch_pbar:
            ray_origins = batch[:, :3].to(device_train)
            ray_directions = batch[:, 3:6].to(device_train)
            ground_truth_px_values = batch[:, 6:].to(device_train)
            
            regenerated_px_values = render_rays(nerf_model, ray_origins, ray_directions, hn=hn, hf=hf, nb_bins=nb_bins) 
            loss = ((ground_truth_px_values - regenerated_px_values) ** 2).sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            current_loss = loss.item()
            training_loss.append(current_loss)
            epoch_loss_sum += current_loss
            num_batches += 1
            
            if num_batches > 0:
                 batch_pbar.set_postfix(loss=f"{current_loss:.4f}", avg_epoch_loss=f"{epoch_loss_sum/num_batches:.4f}")

        scheduler.step()
        avg_epoch_loss = epoch_loss_sum / num_batches if num_batches > 0 else 0
        print(f"Epoch {epoch_idx + 1} completed. Average Loss: {avg_epoch_loss:.4f}")

        # --- Test rendering after each epoch ---
        nerf_model.eval() # Set model to evaluation mode for testing
        if num_images_to_render_epoch > 0:
            print(f"--- Running {num_images_to_render_epoch} test renders for epoch {epoch_idx + 1} ---")
            # Add a tqdm progress bar for the image rendering loop
            for img_idx_test in tqdm(range(num_images_to_render_epoch), desc="Rendering test images", leave=False):
                 test(hn, hf, testing_dataset_for_test_fn, nerf_model, 
                      img_index=img_idx_test, nb_bins=nb_bins, H=H, W=W, chunk_size=10) # Default chunk_size=10
            print(f"--- Test renders for epoch {epoch_idx + 1} complete ---")
            
    return training_loss


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f"Using device: {device}")

    # For dummy data, H_res/W_res are small. For real data, these might be 400x400
    H_data, W_data = 400, 400 # Dimensions of images in your dataset
    

    training_dataset_np = np.load('training_data.pkl', allow_pickle=True)
    testing_dataset_np = np.load('testing_data.pkl', allow_pickle=True)
    training_dataset = torch.from_numpy(training_dataset_np.astype(np.float32))
    testing_dataset = torch.from_numpy(testing_dataset_np.astype(np.float32))
    print("Loaded data from .pkl files.")
        # Infer H, W from loaded data if possible, assuming first image defines it
        # This is a heuristic; better to have these as known parameters.
        # If testing_dataset contains multiple images, and they are all HxW
        # total_rays = testing_dataset.shape[0]
        # num_pixels_per_image_estimate = 400*400 # Default H,W from original test
        # num_images_estimate = total_rays / num_pixels_per_image_estimate
        # print(f"Estimated HxW from test function: {400}x{400}")
        # print(f"Testing dataset shape: {testing_dataset.shape}. Estimated images: {num_images_estimate}")
        # H_data, W_data = 400, 400 # If real data follows the 400x400 convention

    USE_BSPLINE_ENCODING = False
    BSPLINE_POS_MIN_MAX = (-2.5, 2.5) 
    BSPLINE_DIR_MIN_MAX = (-1.0, 1.0) 
    
    hidden_dims_nerf = 256
    learning_rate = 5e-4
    emb_dim_p = 10
    emb_dim_d = 4
    if USE_BSPLINE_ENCODING:
        emb_dim_p = max(2, emb_dim_p) 
        emb_dim_d = max(2, emb_dim_d)

    model = NerfModel(
        embedding_dim_pos=emb_dim_p, embedding_dim_direction=emb_dim_d,
        hidden_dim=hidden_dims_nerf, use_bspline=USE_BSPLINE_ENCODING,
        bspline_pos_min_max=BSPLINE_POS_MIN_MAX, bspline_dir_min_max=BSPLINE_DIR_MIN_MAX
    ).to(device)
    
    model_optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(model_optimizer, milestones=[2, 4, 8], gamma=0.5)
    data_loader_batch_size = 512 
    data_loader = DataLoader(training_dataset, batch_size=data_loader_batch_size, shuffle=True, num_workers=0)
    
    print(f"Training with {'B-spline' if USE_BSPLINE_ENCODING else 'Positional'} Encoding.")
    print(f"NerfModel embedding_dim_pos: {model.embedding_dim_pos}, embedding_dim_direction: {model.embedding_dim_direction}")
    if USE_BSPLINE_ENCODING:
        print(f"  B-spline pos encoder using {model.bspline_pos_encoder.num_bases_per_dim} bases per dim.")
        print(f"  B-spline dir encoder using {model.bspline_dir_encoder.num_bases_per_dim} bases per dim.")

    num_epochs_train = 1 # Set to a low number for quick testing
    
    # Number of novel views to render after each epoch
    # If using dummy data, num_dummy_test_imgs (e.g., 1) will be the max.
    # If using real data, set this to 200 for original behavior, or less for speed.
    NUM_TEST_IMAGES_TO_RENDER_PER_EPOCH = 200 


    near_plane, far_plane = 2., 6.
    num_bins_render = 10
    chunk_size_test_final = 10 # Chunk size for the final test render after all epochs

    training_losses = train(model, model_optimizer, scheduler, data_loader, 
                            testing_dataset_for_test_fn=testing_dataset,
                            num_test_renders_per_epoch=NUM_TEST_IMAGES_TO_RENDER_PER_EPOCH,
                            nb_epochs=num_epochs_train, device_train=device, 
                            hn=near_plane, hf=far_plane, nb_bins=num_bins_render, 
                            H=H_data, W=W_data) # Pass H_data, W_data from your dataset

    print("Training finished.")
    if training_losses:
        plt.figure()
        plt.plot(training_losses) 
        plt.title("Training Loss (per batch)")
        plt.xlabel("Batch Iteration")
        plt.ylabel("Loss")
        plt.savefig("training_loss_per_batch.png")
        # plt.show()
        print("Saved training_loss_per_batch.png")

    # Final test render on one image (example)
    print(f"Running a final test render on image 0 (using H={H_data}, W={W_data})...")
    model.eval() # Ensure model is in eval mode for final test
    test(hn=near_plane, hf=far_plane, dataset=testing_dataset, model_to_test=model, 
         img_index=0, nb_bins=num_bins_render, H=H_data, W=W_data, chunk_size=chunk_size_test_final)
    print("Saved novel_views/img_0.png (final test render)")