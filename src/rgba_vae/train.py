import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

from tqdm import tqdm

from .defaultconfig import DefaultConfig
from .vae import VAE

from typing import List

def vae_loss(recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor) -> List[torch.Tensor]:
    # recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    recon_loss = F.l1_loss(recon_x, x, reduction='sum')

    kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    return recon_loss + kl_divergence, recon_loss, kl_divergence


def save_reconstructions(model: VAE, dataset: torch.utils.data.Dataset, config: DefaultConfig, epoch: int):
    model.eval()
    with torch.no_grad():
        sample_batch, _ = next(iter(DataLoader(dataset, batch_size=8, shuffle=False)))
        x = sample_batch.to(config.device)

        recon_x, _, _ = model(x)

        def denormalize(tensor):
            return tensor * 0.5 + 0.5  # 逆向标准化操作

        original_images = denormalize(x.cpu())
        reconstructed_images = denormalize(recon_x.cpu())

        fig, axes = plt.subplots(2, 8, figsize=(16, 4))
        for i in range(8):
            # original image
            axes[0, i].imshow(original_images[i].permute(1, 2, 0).numpy())
            axes[0, i].axis('off')
            axes[0, i].set_title("Original")

            # reconstructed image
            axes[1, i].imshow(reconstructed_images[i].permute(1, 2, 0).numpy())
            axes[1, i].axis('off')
            axes[1, i].set_title("Reconstructed")

        plt.tight_layout()
        plt.savefig(f"reconstructions/reconstruction_epoch_{epoch}.png")
        plt.close()


def train_vae(vae_model: VAE, dataset: torch.utils.data.Dataset, config: DefaultConfig) -> None:
    vae_model.to(config.device)

    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)

    if config.optimizer == "adamw":
        optimizer = torch.optim.AdamW(vae_model.parameters(), lr=config.learning_rate)
    elif config.optimizer == "adam":
        optimizer = torch.optim.Adam(vae_model.parameters(), lr=config.learning_rate)
    elif config.optimizer == "sgd":
        optimizer = torch.optim.SGD(vae_model.parameters(), lr=config.learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")
    
    for epoch in range(config.num_epochs):
        tqdm.write(f"Epoch {epoch+1}/{config.num_epochs}")
        vae_model.train()
        
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        
        progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch+1}")

        for batch, _ in progress_bar:
            x = batch.to(config.device)

            optimizer.zero_grad()

            recon_x, mu, log_var = vae_model(x)
            loss, recon_loss, kl_loss = vae_loss(recon_x, x, mu, log_var)
            loss.backward()


            optimizer.step()

            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()

        print(f"Epoch [{epoch+1}/{config.num_epochs}] - "
              f"Avg Loss: {total_loss / len(dataloader):.4f}, "
              f"Avg Recon Loss: {total_recon_loss / len(dataloader):.4f}, "
              f"Avg KL Loss: {total_kl_loss / len(dataloader):.4f}")
        
        # save checkpoint
        if (epoch + 1) % config.save_every_n_epochs == 0 or epoch == config.num_epochs - 1:
            checkpoint_path = f"{config.checkpoint_dir}/vae_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': vae_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': f"{total_loss / len(dataloader):.4f}",
            }, checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")

            save_reconstructions(vae_model, dataset, config, epoch + 1)
