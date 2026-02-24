import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from transformers import AutoModel

import matplotlib.pyplot as plt

from tqdm import tqdm

from .defaultconfig import DefaultConfig, VAEGANConfig, VAVAEConfig
from .vae import VAE
from .vaegan import VAEGAN
from .vavae import VAVAE
from .loss import vae_loss, adversarial_loss, feature_matching_loss, vision_alignment_loss


def save_reconstructions(model: VAE, dataset: torch.utils.data.Dataset, config: DefaultConfig, epoch: int):
    model.eval()
    with torch.no_grad():
        sample_batch, _ = next(iter(DataLoader(dataset, batch_size=8, shuffle=False)))
        x = sample_batch.to(config.device)

        recon_x, _, _ = model(x)

        original_images = x.cpu()
        reconstructed_images = recon_x.cpu()

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
        plt.savefig(f"reconstructions/reconstruction_epoch_{epoch}.png", transparent=True)
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

            recon_loss, kl_loss = vae_loss(recon_x, x, mu, log_var)
            loss = config.beta_recon * recon_loss + config.beta_kl * kl_loss
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
    decoder_optimizer = torch.optim.AdamW(vaegan_model.vae.decode.parameters(), lr=config.learning_rate)
    discriminator_optimizer = torch.optim.AdamW(vaegan_model.discriminator.parameters(), lr=config.gan_learning_rate)

    for epoch in range(config.num_epochs):
        tqdm.write(f"Epoch {epoch+1}/{config.num_epochs}")
        vaegan_model.train()

        total_vae_loss = 0
        total_gan_loss = 0
        total_fm_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        total_discriminator_loss = 0

        progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch+1}")

        for batch, _ in progress_bar:
            x = batch.to(config.device)

            # discriminator
            discriminator_optimizer.zero_grad()

            real_validity, _ = vaegan_model.discriminator(x)
        
            with torch.no_grad():
                recon_x, _, _= vaegan_model(x)
            fake_validity, _ = vaegan_model.discriminator(recon_x.detach())
            
            d_loss = adversarial_loss(real_validity, fake_validity)
            d_loss.backward()
            discriminator_optimizer.step()

            #vae (encoder and decoder)
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            recon_x, mu, log_var = vaegan_model(x)

            recon_loss, kl_loss = vae_loss(recon_x, x, mu, log_var)

            v_loss = config.beta_recon * recon_loss + config.beta_kl * kl_loss

            _, real_features = vaegan_model.discriminator(x)

            fake_validity_gen, fake_features_gen = vaegan_model.discriminator(recon_x)

            # gan loss
            g_loss = F.binary_cross_entropy(fake_validity_gen, torch.ones_like(fake_validity_gen))


            # feature matching loss
            fm_loss = feature_matching_loss(real_features, fake_features_gen)

            generator_loss =  v_loss + config.beta_g*g_loss + config.beta_fm*fm_loss 

            generator_loss.backward()

            encoder_optimizer.step()
            decoder_optimizer.step()

            total_vae_loss += v_loss.item()
            total_gan_loss += g_loss.item()
            total_fm_loss += fm_loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            total_discriminator_loss += d_loss.item()

        print(
            f"Epoch [{epoch+1}/{config.num_epochs}] - "
            f"x range:{recon_x.min():.4f} - {recon_x.max():.4f}"
            f"Avg VAE Loss: {total_vae_loss / len(dataloader):.4f}, "
            f"Avg GAN Loss: {total_gan_loss / len(dataloader):.4f}, "
            f"Avg FM Loss: {total_fm_loss / len(dataloader):.4f},"
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
                        "fm_loss": total_fm_loss / len(dataloader),
                        "recon_loss": total_recon_loss / len(dataloader),
                        "kl_loss": total_kl_loss / len(dataloader),
                        "discriminator_loss": total_discriminator_loss / len(dataloader),
                    },
                },
                checkpoint_path,
            )
            print(f"Checkpoint saved at {checkpoint_path}")

            save_reconstructions(vaegan_model.vae, dataset, config, epoch + 1)


def train_vavae(vae_model: VAVAE, dataset: torch.utils.data.Dataset, config: VAVAEConfig):
    vae_model.to(config.device)

    vf_model = AutoModel.from_pretrained(config.vf_model_name)
    vf_model.eval().to(config.device)

    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)

    vae_optimizer = torch.optim.AdamW(vae_model.vae.parameters(), lr=config.learning_rate)
    vision_align_optimizer = torch.optim.AdamW(vae_model.visionalignment.parameters(), lr=config.align_learning_rate)

    if config.use_deterministic:
        discriminator_optimizer = torch.optim.AdamW(vae_model.discriminator.parameters(), lr=config.gan_learning_rate)

    for epoch in range(config.num_epochs):
        tqdm.write(f"Epoch {epoch+1}/{config.num_epochs}")
        vae_model.train()

        if config.use_deterministic:
            total_gan_loss = 0
            total_fm_loss = 0
            total_discriminator_loss = 0

        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        total_vf_loss = 0

        progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch+1}")

        for batch, _ in progress_bar:
            x = batch.to(config.device)

            # if discriminator avaliable
            if config.use_deterministic:
                # Discriminator training
                discriminator_optimizer.zero_grad()

                real_validity, _ = vae_model.discriminator(x)

                with torch.no_grad():
                    recon_x, _, _ = vae_model(x)
                fake_validity, _ = vae_model.discriminator(recon_x.detach())

                d_loss = adversarial_loss(real_validity, fake_validity)
                d_loss.backward()
                discriminator_optimizer.step()

                total_discriminator_loss += d_loss.item()

            # vae and vision alignment
            vae_optimizer.zero_grad()
            vision_align_optimizer.zero_grad()

            recon_x, mu, log_var = vae_model(x)

            recon_loss, kl_loss = vae_loss(recon_x, x, mu, log_var)

            # vision alignment loss
            with torch.no_grad():
                x_rgb = x[:, :3, :, :]
                vf_outputs = vf_model(x_rgb)
                vf_features = vf_outputs.last_hidden_state.mean(dim=1)
            
            
            vf_sim = vae_model.visionalignment(mu, log_var, vf_features)
            vf_loss = vision_alignment_loss(vf_sim)

            loss = config.beta_recon * recon_loss + config.beta_kl * kl_loss + config.beta_vf * vf_loss

            if config.use_deterministic:
                _, real_features = vae_model.discriminator(x)
                fake_validity_gen, fake_features_gen = vae_model.discriminator(recon_x)

                # GAN loss
                g_loss = F.binary_cross_entropy(fake_validity_gen, torch.ones_like(fake_validity_gen))

                # Feature matching loss
                fm_loss = feature_matching_loss(real_features, fake_features_gen)

                generator_loss = loss + config.beta_g*g_loss + config.beta_fm*fm_loss
                generator_loss.backward()

                total_gan_loss += g_loss.item()
                total_fm_loss += fm_loss.item()
            else:
                loss.backward()

            vae_optimizer.step()
            vision_align_optimizer.step()

            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            total_vf_loss += vf_loss.item()
        
        # print training log
        if config.use_deterministic:
            print(
                f"Epoch [{epoch+1}/{config.num_epochs}] - "
                f"Avg Loss: {total_loss / len(dataloader):.4f}, "
                f"Avg Recon Loss: {total_recon_loss / len(dataloader):.4f}, "
                f"Avg KL Loss: {total_kl_loss / len(dataloader):.4f}, "
                f"Avg VF Loss: {total_vf_loss / len(dataloader):.4f}, "
                f"Avg GAN Loss: {total_gan_loss / len(dataloader):.4f}, "
                f"Avg FM Loss: {total_fm_loss / len(dataloader):.4f}, "
                f"Avg Discriminator Loss: {total_discriminator_loss / len(dataloader):.4f}"
            )
        else:
            print(
                f"Epoch [{epoch+1}/{config.num_epochs}] - "
                f"Avg Loss: {total_loss / len(dataloader):.4f}, "
                f"Avg Recon Loss: {total_recon_loss / len(dataloader):.4f}, "
                f"Avg KL Loss: {total_kl_loss / len(dataloader):.4f}, "
                f"Avg VF Loss: {total_vf_loss / len(dataloader):.4f}"
            )
            

        # save checkpoint
        if (epoch + 1) % config.save_every_n_epochs == 0 or epoch == config.num_epochs - 1:
            checkpoint_path = f"{config.checkpoint_dir}/vavae_epoch_{epoch+1}.pth"
            state_dict = {
                "epoch": epoch,
                "vae_state_dict": vae_model.vae.state_dict(),
                "vision_align_state_dict": vae_model.visionalignment.state_dict(),
                "vae_optimizer_state_dict": vae_optimizer.state_dict(),
                "vision_align_optimizer_state_dict": vision_align_optimizer.state_dict(),
                "loss": {
                    "total_loss": total_loss / len(dataloader),
                    "recon_loss": total_recon_loss / len(dataloader),
                    "kl_loss": total_kl_loss / len(dataloader),
                    "vf_loss": total_vf_loss / len(dataloader),
                },
            }

            if config.use_deterministic:
                state_dict["discriminator_state_dict"] = vae_model.discriminator.state_dict()
                state_dict["discriminator_optimizer_state_dict"] = discriminator_optimizer.state_dict()
                state_dict["loss"]["gan_loss"] = total_gan_loss / len(dataloader)
                state_dict["loss"]["fm_loss"] = total_fm_loss / len(dataloader)
                state_dict["loss"]["discriminator_loss"] = total_discriminator_loss / len(dataloader)

            torch.save(state_dict, checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")

            save_reconstructions(vae_model.vae, dataset, config, epoch + 1)
