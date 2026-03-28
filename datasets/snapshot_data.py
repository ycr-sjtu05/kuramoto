import os
import torch
from torch.utils.data import Dataset, DataLoader

class SnapshotDataset(Dataset):
    """
    专门用于读取离线生成的 Kuramoto 物理快照的极速数据加载器。
    它不处理任何图像解码或 Augmentation，只负责极致的 I/O 读取。
    """
    def __init__(self, snapshot_dir, return_index=False):
        super().__init__()
        self.snapshot_dir = snapshot_dir
        self.return_index = return_index

        # 扫描目录下所有的 .pt 文件并按字典序排序
        self.file_names = sorted([f for f in os.listdir(snapshot_dir) if f.endswith('.pt')])

        if len(self.file_names) == 0:
            raise ValueError(
                f"在 {snapshot_dir} 目录下没有找到任何 .pt 文件！\n"
                f"请务必先运行 scripts/generate_snapshots.py 生成离线快照。"
            )

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_path = os.path.join(self.snapshot_dir, self.file_names[idx])

        # 直接使用 torch.load 加载张量
        snapshots = torch.load(file_path, weights_only=True)

        # snapshots shape: (N_snapshots, C, H, W)
        if self.return_index:
            return snapshots, idx
        return snapshots


#=========================
# 测试代码 (可以独立运行本文件测试)
# ==========================================
if __name__ == "__main__":
    # 假设你之前生成的快照存在这个目录下
    test_dir = "../data/snapshots/socofing_kura_200_11"
    
    # 造一个假的目录和文件用于测试逻辑 (如果你还没跑预处理的话)
    os.makedirs(test_dir, exist_ok=True)
    fake_tensor = torch.randn(11, 1, 96, 96) # 11张快照，单通道，96x96
    torch.save(fake_tensor, os.path.join(test_dir, "000000.pt"))
    torch.save(fake_tensor, os.path.join(test_dir, "000001.pt"))
    
    # 实例化我们的 Dataset
    dataset = SnapshotDataset(snapshot_dir=test_dir)
    print(f"成功加载数据集，共找到 {len(dataset)} 个样本。")
    
    # 放入 DataLoader (设置 num_workers > 0 可以极大加速 I/O)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2)
    
    for batch_idx, batch_snapshots in enumerate(dataloader):
        # 此时的 batch_snapshots 形状应当是 (Batch, N_snapshots, C, H, W)
        print(f"Batch {batch_idx} 读取成功! Tensor Shape: {batch_snapshots.shape}")
        break