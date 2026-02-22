import torch
from rgba_vae import train_vaegan, VAEGAN, VAEGANConfig
from rgba_vae import JSONLBase64Dataset

def main():
    # 1. 加载配置
    config = VAEGANConfig(image_size=128)
    
    # 2. 创建数据集
    dataset = JSONLBase64Dataset(
        jsonl_file="I:/myproject2026/rgba-vae/cover/dataset_0000.jsonl_slim.jsonl",
    )
    
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