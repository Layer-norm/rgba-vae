import torch
from rgba_vae import train_vae, VAE, DefaultConfig
from toydataset import ToyTextImageDataset

def main():
    # 1. 加载配置
    config = DefaultConfig()
    
    # 2. 创建数据集
    dataset = ToyTextImageDataset(num_samples=1000)
    
    # 3. 初始化VAE模型
    vae_model = VAE(
        in_channels=config.in_channels,
        image_size=config.image_size,
        hidden_dims=config.hidden_dims,
        latent_dim=config.latent_dim,
        dropout=config.dropout
    )
    
    # 4. 开始训练
    train_vae(vae_model, dataset, config)

if __name__ == "__main__":
    main()