"""
用于DDP的双流批量采样器
包括了测试代码
"""

import torch
import random
from collections import defaultdict

import torch
import random
import math # 导入 math 库用于向上取整
from collections import defaultdict
from torch.utils.data import Sampler # 继承自Sampler基类，更规范

class DDPBalancedBatchSampler(Sampler):
    """
    支持DDP的、经过修正的平衡双流采样器。
    - 解决了数据丢失问题。
    - 修正了 __len__ 方法。
    - 保证了每个副本在每个epoch处理相同数量的批次。
    """
    def __init__(self, primary_indices, secondary_indices, primary_bs, secondary_bs,
                 multiplier, num_replicas=None, rank=None, shuffle=True, seed=0):
        
        # FIX: 自动获取DDP环境信息
        if num_replicas is None:
            if not torch.distributed.is_available() or not torch.distributed.is_initialized():
                num_replicas = 1
            else:
                num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available() or not torch.distributed.is_initialized():
                rank = 0
            else:
                rank = torch.distributed.get_rank()

        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.primary_bs = primary_bs
        self.secondary_bs = secondary_bs
        self.multiplier = multiplier
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        # FIX 1: 修正长度计算逻辑，不再提前划分数据
        # 计算每个副本需要处理的样本数，通过向上取整和填充保证数据不丢失
        self.num_primary_samples = math.ceil(len(self.primary_indices) / self.num_replicas)
        self.total_primary_size = self.num_primary_samples * self.num_replicas
        
        self.num_secondary_samples = math.ceil(len(self.secondary_indices) / self.num_replicas)
        self.total_secondary_size = self.num_secondary_samples * self.num_replicas
        
    def set_epoch(self, epoch: int) -> None:
        """设置epoch（用于shuffle）"""
        self.epoch = epoch
        
    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        
        # --- 有标签数据处理 ---
        primary_indices = self.primary_indices
        if self.shuffle:
            perm = torch.randperm(len(primary_indices), generator=g).tolist()
            primary_indices = [primary_indices[i] for i in perm]
        
        # FIX 1: 填充数据集以确保可以被 num_replicas 整除
        padding_size = self.total_primary_size - len(primary_indices)
        if padding_size > 0:
            primary_indices += primary_indices[:padding_size]
        
        # 为当前rank选择子集
        primary_indices = primary_indices[self.rank:self.total_primary_size:self.num_replicas]
        
        # --- 无标签数据处理 ---
        secondary_indices = self.secondary_indices
        if self.shuffle:
            perm = torch.randperm(len(secondary_indices), generator=g).tolist()
            secondary_indices = [secondary_indices[i] for i in perm]
            
        # 填充数据集
        padding_size = self.total_secondary_size - len(secondary_indices)
        if padding_size > 0:
            secondary_indices += secondary_indices[:padding_size]
            
        # 为当前rank选择子集
        secondary_indices = secondary_indices[self.rank:self.total_secondary_size:self.num_replicas]

        # 扩展无标签数据以满足倍数要求
        extended_secondary = []
        # 为了避免内存爆炸，如果multiplier很大，我们循环使用无标签索引
        num_primary_batches = math.ceil(len(primary_indices) / self.primary_bs)
        required_secondary_samples = num_primary_batches * self.secondary_bs
        
        # 使用 itertools.cycle 的思想来高效重复
        from itertools import cycle
        secondary_iterator = cycle(secondary_indices)
        extended_secondary = [next(secondary_iterator) for _ in range(required_secondary_samples)]

        # 生成批次
        primary_batches = [primary_indices[i:i+self.primary_bs] 
                           for i in range(0, len(primary_indices), self.primary_bs)]
        secondary_batches = [extended_secondary[i:i+self.secondary_bs] 
                             for i in range(0, len(extended_secondary), self.secondary_bs)]

        # 合并批次并返回
        for p_batch, s_batch in zip(primary_batches, secondary_batches):
            yield p_batch + s_batch
            
    def __len__(self) -> int:
        # FIX 2: 修正长度计算
        # 长度由有标签数据的批次数决定
        return math.ceil(self.num_primary_samples / self.primary_bs)

# --- 修改后的测试函数 ---
def test_sampler(config, shuffle=False):
    """测试采样器功能"""
    # 解析配置
    primary_indices = config['primary_indices']
    secondary_indices = config['secondary_indices']
    primary_bs = config['primary_bs']
    secondary_bs = config['secondary_bs']
    multiplier = config['multiplier'] # multiplier不再直接用于__len__
    num_replicas = config['num_replicas']
    seed = config.get('seed', 42)
    
    print(f"\n\n{'='*50}")
    print(f"测试配置: {config}")
    print(f"{'='*50}")
    
    samplers = []
    for rank in range(num_replicas):
        sampler = DDPBalancedBatchSampler(
            primary_indices, secondary_indices, primary_bs, secondary_bs,
            multiplier, num_replicas, rank, shuffle=shuffle, seed=seed
        )
        samplers.append(sampler)
        print(f"\n===== Rank {rank} 采样器信息 =====")
        print(f"每个副本的有标签样本数: {sampler.num_primary_samples}")
        print(f"每个副本的无标签样本数: {sampler.num_secondary_samples}")
        print(f"总批次数 (__len__): {len(sampler)}")

    all_used_indices = defaultdict(int)
    num_epochs = 2
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        
        for sampler in samplers:
            sampler.set_epoch(epoch)
        
        for rank, sampler in enumerate(samplers):
            batches = list(sampler)
            print(f"\nRank {rank} (共 {len(batches)} 个批次):")
            
            for i, batch in enumerate(batches):
                # FIX 3: 使用正确的 bs 来拆分批次
                primary_samples = batch[:primary_bs]
                secondary_samples = batch[primary_bs:]
                
                # 检查批次大小是否正确 (最后一个批次可能不完整)
                assert len(primary_samples) <= primary_bs
                assert len(secondary_samples) <= secondary_bs
                
                for idx in batch:
                    if idx in primary_indices:
                        all_used_indices[('primary', idx)] += 1
                    elif idx in secondary_indices:
                        all_used_indices[('secondary', idx)] += 1
                
                print(f"  批次 {i+1}:")
                print(f"    有标签 (bs={len(primary_samples)}): {primary_samples}")
                print(f"    无标签 (bs={len(secondary_samples)}): {secondary_samples}")
    
    print("\n样本使用统计 (2个Epochs):")
    for (data_type, idx), count in sorted(all_used_indices.items()):
        print(f"  {data_type.capitalize()}样本 {idx}: 使用 {count} 次")
        if data_type == 'primary':
            # 每个主样本应该被使用 num_epochs * (可能被填充的次数)
            assert count > 0
        elif data_type == 'secondary':
            # 无标签样本的使用次数会更多
            assert count > 0

def print_sampler_info(sampler, rank):
    """打印采样器信息"""
    print(f"\n===== Rank {rank} 采样器信息 =====")
    print(f"有标签数据范围: {sampler.primary_start}-{sampler.primary_end}")
    print(f"无标签数据范围: {sampler.secondary_start}-{sampler.secondary_end}")
    print(f"有标签数据样本: {sampler.primary_indices[sampler.primary_start:sampler.primary_end]}")
    print(f"无标签数据样本: {sampler.secondary_indices[sampler.secondary_start:sampler.secondary_end]}")
    print(f"总批次数: {len(sampler)}")

def main():
    """主测试函数"""
    # 测试配置1: 基础配置 (2个GPU)
    config1 = {
        'primary_indices': list(range(10)),  # 10个有标签样本
        'secondary_indices': list(range(10, 30)),  # 20个无标签样本
        'primary_bs': 2,  # 每个批次的有标签样本数
        'secondary_bs': 3,  # 每个批次的无标签样本数
        'multiplier': 2,  # 无标签数据使用倍数
        'num_replicas': 2,  # GPU数量
        'seed': 42
    }
    
    # 测试配置2: 样本数不能整除 (3个GPU)
    config2 = {
        'primary_indices': list(range(7)),  # 7个有标签样本
        'secondary_indices': list(range(7, 22)),  # 15个无标签样本
        'primary_bs': 1,  # 每个批次的有标签样本数
        'secondary_bs': 2,  # 每个批次的无标签样本数
        'multiplier': 3,  # 无标签数据使用倍数
        'num_replicas': 3,  # GPU数量
        'seed': 123
    }
    
    # 测试配置3: 大倍数配置 (4个GPU)
    config3 = {
        'primary_indices': list(range(8)),  # 8个有标签样本
        'secondary_indices': list(range(8, 24)),  # 16个无标签样本
        'primary_bs': 2,  # 每个批次的有标签样本数
        'secondary_bs': 2,  # 每个批次的无标签样本数
        'multiplier': 4,  # 无标签数据使用倍数
        'num_replicas': 4,  # GPU数量
        'seed': 456
    }
    
    # 运行测试 (关闭shuffle以便观察数据分布)
    print("===== 测试1: 基础配置 (无shuffle) =====")
    test_sampler(config1, shuffle=False)
    
    print("\n\n===== 测试2: 样本数不能整除 (无shuffle) =====")
    test_sampler(config2, shuffle=False)
    
    print("\n\n===== 测试3: 大倍数配置 (无shuffle) =====")
    test_sampler(config3, shuffle=False)
    
    # 测试shuffle功能
    print("\n\n===== 测试4: 基础配置 (启用shuffle) =====")
    test_sampler(config1, shuffle=True)

if __name__ == "__main__":
    main()