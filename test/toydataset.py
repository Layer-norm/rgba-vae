# ====================
# RGBA Toy Dataset for Training
# Generate by Qwen (Modified for transparent background)
# ====================

from PIL import Image, ImageDraw
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class ToyTextImageDataset(Dataset):
    """增强版RGBA玩具数据集, 支持透明背景和随机颜色形状组合"""
    def __init__(self, num_samples=1000):
        self.num_samples = num_samples
        # 修改变换以适应RGBA图像
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])
        
        # 定义颜色和形状词汇库，用于随机组合
        self.colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 
                      'black', 'white', 'gray', 'brown', 'pink', 'violet']
        self.shapes = ['circle', 'square', 'triangle', 'star', 'heart', 'diamond',
                      'cross', 'pentagon', 'hexagon', 'oval', 'rectangle']
        
        # 颜色映射 - 现在包括alpha通道
        self.color_map = {
            'red': (255, 0, 0, 255), 'blue': (0, 0, 255, 255), 'green': (0, 255, 0, 255),
            'yellow': (255, 255, 0, 255), 'purple': (147, 112, 219, 255), 'orange': (255, 165, 0, 255),
            'black': (0, 0, 0, 255), 'white': (255, 255, 255, 255), 'gray': (128, 128, 128, 255),
            'brown': (205, 127, 50, 255), 'pink': (255, 192, 203, 255), 'violet': (238, 130, 238, 255)
        }
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # 随机选择颜色和形状
        color = random.choice(self.colors)
        shape = random.choice(self.shapes)
        prompt = f"a {color} {shape}"
        
        # 生成RGBA图像
        image = self.generate_rgba_image(color, shape)
        
        return self.transform(image), prompt
    
    def generate_rgba_image(self, color, shape):
        """生成带有透明背景的RGBA图像"""
        # 创建RGBA模式的图像，初始为完全透明
        img = Image.new('RGBA', (64, 64), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # 获取颜色值（现在包含alpha通道）
        fill_color = self.color_map.get(color, (0, 0, 0, 255))
        
        # 如果是白色，稍微调整使其可见于透明背景
        if color == 'white':
            fill_color = (245, 245, 245, 255)
        
        # 随机确定位置和大小，增加多样性
        margin = 10
        max_size = 44  # 64 - 2*margin
        
        # 随机决定形状的位置和大小
        pos_x = random.randint(margin, 64 - margin - max_size//2)
        pos_y = random.randint(margin, 64 - margin - max_size//2)
        size_factor = random.uniform(0.7, 1.0)  # 大小变化因子
        actual_size = int(max_size * size_factor)
        
        end_x = pos_x + actual_size
        end_y = pos_y + actual_size
        
        # 绘制不同形状
        if shape == 'circle':
            draw.ellipse([pos_x, pos_y, end_x, end_y], fill=fill_color)
        elif shape == 'square':
            draw.rectangle([pos_x, pos_y, end_x, end_y], fill=fill_color)
        elif shape == 'triangle':
            center_x = (pos_x + end_x) // 2
            center_y = (pos_y + end_y) // 2
            half_size = actual_size // 2
            points = [
                (center_x, pos_y),           # 顶点
                (end_x, end_y),              # 右下角
                (pos_x, end_y)               # 左下角
            ]
            draw.polygon(points, fill=fill_color)
        elif shape == 'star':
            self._draw_star(draw, pos_x, pos_y, actual_size, fill_color)
        elif shape == 'heart':
            self._draw_heart(draw, pos_x, pos_y, actual_size, fill_color)
        elif shape == 'diamond':
            center_x = (pos_x + end_x) // 2
            center_y = (pos_y + end_y) // 2
            points = [
                (center_x, pos_y),           # 上
                (end_x, center_y),           # 右
                (center_x, end_y),           # 下
                (pos_x, center_y)            # 左
            ]
            draw.polygon(points, fill=fill_color)
        elif shape == 'cross':
            center_x = (pos_x + end_x) // 2
            center_y = (pos_y + end_y) // 2
            width = actual_size // 5
            # 水平线
            draw.rectangle([pos_x, center_y - width//2, end_x, center_y + width//2], fill=fill_color)
            # 垂直线
            draw.rectangle([center_x - width//2, pos_y, center_x + width//2, end_y], fill=fill_color)
        elif shape == 'pentagon':
            self._draw_polygon(draw, pos_x, pos_y, actual_size, 5, fill_color)
        elif shape == 'hexagon':
            self._draw_polygon(draw, pos_x, pos_y, actual_size, 6, fill_color)
        elif shape == 'oval':
            aspect_ratio = random.uniform(0.6, 1.0)
            adjusted_width = actual_size
            adjusted_height = int(actual_size * aspect_ratio)
            offset_y = (actual_size - adjusted_height) // 2
            draw.ellipse([pos_x, pos_y + offset_y, end_x, pos_y + offset_y + adjusted_height], fill=fill_color)
        elif shape == 'rectangle':
            aspect_ratio = random.uniform(0.5, 1.5)  # 长宽比
            if random.random() > 0.5:
                # 水平矩形
                adjusted_width = min(actual_size, int(actual_size * aspect_ratio))
                adjusted_height = actual_size
            else:
                # 竖直矩形
                adjusted_width = actual_size
                adjusted_height = min(actual_size, int(actual_size * aspect_ratio))
            
            center_x = (pos_x + end_x) // 2
            center_y = (pos_y + end_y) // 2
            
            start_x = center_x - adjusted_width // 2
            start_y = center_y - adjusted_height // 2
            end_rect_x = start_x + adjusted_width
            end_rect_y = start_y + adjusted_height
            
            draw.rectangle([start_x, start_y, end_rect_x, end_rect_y], fill=fill_color)
        else:
            # 默认画圆
            draw.ellipse([pos_x, pos_y, end_x, end_y], fill=fill_color)
        
        return img
    
    def _draw_star(self, draw, pos_x, pos_y, size, color):
        """绘制五角星"""
        center_x = pos_x + size // 2
        center_y = pos_y + size // 2
        outer_radius = size // 2
        inner_radius = outer_radius // 2
        
        points = []
        for i in range(10):
            angle = 2 * np.pi * i / 10 - np.pi / 2
            radius = outer_radius if i % 2 == 0 else inner_radius
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)
            points.append((x, y))
        
        draw.polygon(points, fill=color)
    
    def _draw_heart(self, draw, pos_x, pos_y, size, color):
        """绘制心形"""
        # 简化的心形绘制，使用椭圆组合
        center_x = pos_x + size // 2
        center_y = pos_y + size // 2
        radius = size // 4
        
        # 左半圆
        draw.ellipse([center_x - radius, pos_y + radius//2, center_x, center_y + radius//2], fill=color)
        # 右半圆
        draw.ellipse([center_x, pos_y + radius//2, center_x + radius, center_y + radius//2], fill=color)
        # 底部三角
        points = [(center_x - radius, center_y), (center_x + radius, center_y), (center_x, center_y + radius)]
        draw.polygon(points, fill=color)
    
    def _draw_polygon(self, draw, pos_x, pos_y, size, sides, color):
        """绘制多边形"""
        center_x = pos_x + size // 2
        center_y = pos_y + size // 2
        radius = size // 2
        
        points = []
        for i in range(sides):
            angle = 2 * np.pi * i / sides - np.pi / 2  # 从顶部开始
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)
            points.append((x, y))
        
        draw.polygon(points, fill=color)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # 创建数据集实例
    dataset = ToyTextImageDataset(num_samples=1000)
    
    print(f"数据集大小: {len(dataset)}")
    print(f"颜色种类 ({len(dataset.colors)}): {dataset.colors}")
    print(f"形状种类 ({len(dataset.shapes)}): {dataset.shapes}")
    
    # 测试几个样本
    print("\n前5个样本的文本提示:")
    for i in range(5):
        sample = dataset[i]
        print(f"样本 {i+1}: {sample['text']}")
    
    # 可视化前12个样本
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    axes = axes.ravel()
    
    for i in range(min(12, len(dataset))):
        sample = dataset[i]
        image = sample['image']
        text = sample['text']
        
        # 将张量转换回PIL图像以便显示
        # 转换为numpy数组并调整维度顺序 (C, H, W) -> (H, W, C)
        image_np = image.permute(1, 2, 0).numpy()
        # 确保数值在[0, 1]范围内
        image_np = np.clip(image_np, 0, 1)
        
        axes[i].imshow(image_np)
        axes[i].set_title(text, fontsize=10)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # 检查数据维度是否为4通道
    sample = dataset[0]
    print(f"\n图像张量形状: {sample['image'].shape}")
    print("预期形状: [4, 64, 64] (RGBA, Height, Width)")
    
    # 统计生成样本的一些特征
    print(f"\n随机生成的样本示例:")
    for i in range(10):
        sample = dataset[random.randint(0, len(dataset)-1)]
        print(f"  {sample['text']}")
    
    # 检查是否所有颜色和形状都被使用过（通过多次采样验证多样性）
    used_colors = set()
    used_shapes = set()
    test_samples = 50  # 测试50个样本来检查覆盖率
    
    for i in range(test_samples):
        idx = random.randint(0, len(dataset)-1)
        sample = dataset[idx]
        text = sample['text']
        # 提取颜色和形状
        for color in dataset.colors:
            if color in text:
                used_colors.add(color)
                break
        for shape in dataset.shapes:
            if shape in text:
                used_shapes.add(shape)
                break
    
    print(f"\n在{test_samples}个随机样本中:")
    print(f"使用的颜色数量: {len(used_colors)}/{len(dataset.colors)}")
    print(f"使用的形状数量: {len(used_shapes)}/{len(dataset.shapes)}")
    print(f"颜色覆盖率: {len(used_colors)/len(dataset.colors)*100:.1f}%")
    print(f"形状覆盖率: {len(used_shapes)/len(dataset.shapes)*100:.1f}%")