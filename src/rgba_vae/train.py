import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

from tqdm import tqdm

from .defaultconfig import DefaultConfig, VAEGANConfig
from .vae import VAE
from .vaegan import VAEGAN

from typing import List

def vae_loss(recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor) -> List[torch.Tensor]:
    # recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    recon_loss = F.l1_loss(recon_x, x, reduction='sum')

    kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    return recon_loss + kl_divergence, recon_loss, kl_divergence


def adversarial_loss(
        real_validity: torch.Tensor,
        fake_validity: torch.Tensor
    ) -> List[torch.Tensor]:


    adversarial_loss = F.binary_cross_entropy(real_validity, torch.ones_like(real_validity)) + \
               F.binary_cross_entropy(fake_validity, torch.zeros_like(fake_validity))

    return adversarial_loss



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


def train_vaegan(vaegan_model: VAEGAN, dataset: torch.utils.data.Dataset, config: VAEGANConfig) -> None:
    vaegan_model.to(config.device)

    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    
    encoder_optimizer = torch.optim.AdamW(vaegan_model.vae.encode.parameters(), lr=config.learning_rate)
    decoder_optimizer = torch.optim.AdamW(vaegan_model.vae.dncode.parameters(), lr=config.learning_rate)
    discriminator_optimizer = torch.optim.AdamW(vaegan_model.discriminator.parameters(), lr=config.learning_rate)

    for epoch in range(config.num_epochs):
        tqdm.write(f"Epoch {epoch+1}/{config.num_epochs}")
        vaegan_model.train()

        total_vae_loss = 0
        total_gan_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        total_discriminator_loss = 0

        progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch+1}")

        for batch, _ in progress_bar:
            x = batch.to(config.device)

            # discriminator
            real_validity = vaegan_model.discriminator(x)
        
            with torch.no_grad():
                _, _, _, real_validity, fake_validity = vaegan_model(x)
            
            d_loss = adversarial_loss(real_validity, fake_validity)
            d_loss.backward()
            discriminator_optimizer.step()

            #vae (encoder and decoder)
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            recon_x, mu, log_var, _, _ = vaegan_model(x)

            v_loss, recon_loss, kl_loss = vae_loss(recon_x, x, mu, log_var)

            fake_gan_validity = vaegan_model.discriminator(recon_x)
            g_loss = F.binary_cross_entropy(fake_validity, torch.ones_like(fake_validity))

            generator_loss =  v_loss + g_loss

            generator_loss.backward()

            encoder_optimizer.step()
            decoder_optimizer.step()

            total_vae_loss += v_loss.item()
            total_gan_loss += g_loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            total_discriminator_loss += d_loss.item()

        print(
            f"Epoch [{epoch+1}/{config.num_epochs}] - "
            f"Avg VAE Loss: {total_vae_loss / len(dataloader):.4f}, "
            f"Avg GAN Loss: {total_gan_loss / len(dataloader):.4f}, "
            f"Avg Recon Loss: {total_recon_loss / len(dataloader):.4f}, "
            f"Avg KL Loss: {total_kl_loss / len(dataloader):.4f}, "
            f"Avg Discriminator Loss: {total_discriminator_loss / len(dataloader):.4f}"
        )

        # save checkpoint
        if (epoch + 1) % config.save_every_n_epochs == 0 or epoch == config.num_epochs - 1:
            checkpoint_path = f"{config.checkpoint_dir}/vaegan_epoch_{epoch+1}.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "vae_state_dict": vaegan_model.vae.state_dict(),
                    "discriminator_state_dict": vaegan_model.discriminator.state_dict(),
                    "encoder_optimizer_state_dict": encoder_optimizer.state_dict(),
                    "decoder_optimizer_state_dict": decoder_optimizer.state_dict(),
                    "discriminator_optimizer_state_dict": discriminator_optimizer.state_dict(),
                    "loss": {
                        "vae_loss": total_vae_loss / len(dataloader),
                        "gan_loss": total_gan_loss / len(dataloader),
                        "recon_loss": total_recon_loss / len(dataloader),
                        "kl_loss": total_kl_loss / len(dataloader),
                        "discriminator_loss": total_discriminator_loss / len(dataloader),
                    },
                },
                checkpoint_path,
            )
            print(f"Checkpoint saved at {checkpoint_path}")

            save_reconstructions(vaegan_model.vae, dataset, config, epoch + 1)