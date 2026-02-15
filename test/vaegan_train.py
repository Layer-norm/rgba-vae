import torch
from rgba_vae import train_vaegan, VAEGAN, VAEGANConfig
from toydataset import ToyTextImageDataset

def main():
    # 1. 加载配置
    config = VAEGANConfig()
    
    # 2. 创建数据集
    dataset = ToyTextImageDataset(num_samples=1000)
    
    # 3. 初始化VAE模型
    vae_model = VAEGAN(
        in_channels=config.in_channels,
        image_size=config.image_size,
        hidden_dims=config.hidden_dims,
        latent_dim=config.latent_dim,
        dropout=config.dropout
    )

    # 4. 开始训练
    train_vaegan(vae_model, dataset, config)

if __name__ == "__main__":
    main()