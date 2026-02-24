import torch
from rgba_vae import train_vavae, VAVAE, VAVAEConfig
from toydataset import ToyTextImageDataset

def main():
    # 1. 加载配置
    config = VAVAEConfig(
        use_deterministic=True,
        hidden_dims=[96, 192, 384, 768],
        latent_dim=512,
    )
    
    # 2. 创建数据集
    dataset = ToyTextImageDataset(num_samples=2000)
    
    # 3. 初始化VAE模型
    vae_model = VAVAE(
        in_channels=config.in_channels,
        image_size=config.image_size,
        hidden_dims=config.hidden_dims,
        latent_dim=config.latent_dim,
        vf_feature_dim=config.vf_feature_dim,
        dropout=config.dropout
    )

    # 4. 开始训练
    train_vavae(vae_model, dataset, config)

if __name__ == "__main__":
    main()