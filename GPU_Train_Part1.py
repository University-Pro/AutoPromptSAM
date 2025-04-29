import torch
import time
import argparse

def occupy_gpu_memory(gpu_id: int, mb_to_occupy: int, hold_seconds: int = 0):
    """
    占用指定GPU的显存并保持不释放
    :param gpu_id: 要占用的GPU编号 (如0表示GPU0)
    :param mb_to_occupy: 要占用的显存大小(MB)
    :param hold_seconds: 保持占用的时间(秒)，0表示永久占用
    """
    assert torch.cuda.is_available(), "CUDA不可用"
    assert gpu_id < torch.cuda.device_count(), f"GPU{gpu_id}不存在"
    
    device = torch.device(f'cuda:{gpu_id}')
    torch.cuda.set_device(device)
    
    bytes_to_occupy = mb_to_occupy * 1024 * 1024  # 转换为字节
    elements = bytes_to_occupy // 4  # float32占4字节
    
    # 创建显存占用张量
    try:
        dummy_tensor = torch.randn(elements, dtype=torch.float32, device=device)
        torch.cuda.synchronize()  # 确保显存已分配
        
        # 打印实际占用情况
        allocated = torch.cuda.memory_allocated(device) / 1024**2
        reserved = torch.cuda.memory_reserved(device) / 1024**2
        print(f"GPU{gpu_id}显存占用成功 | "
              f"实际占用: {allocated:.2f}MB | "
              f"保留显存: {reserved:.2f}MB")
        
        # 保持占用状态
        if hold_seconds > 0:
            print(f"将在{hold_seconds}秒后释放...")
            time.sleep(hold_seconds)
        else:
            print("持续占用中 (Ctrl+C 终止)...")
            while True:
                time.sleep(3600)  # 永久阻塞
                
    except RuntimeError as e:
        print(f"显存分配失败: {str(e)}")
    except KeyboardInterrupt:
        print("用户中断，释放显存")
    finally:
        if 'dummy_tensor' in locals():
            del dummy_tensor
        torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU显存占用工具")
    parser.add_argument("--gpu", type=int, required=True, help="要占用的GPU编号")
    parser.add_argument("--mb", type=int, required=True, help="要占用的显存大小(MB)")
    parser.add_argument("--hold", type=int, default=0, help="占用持续时间(秒)，0表示永久")
    args = parser.parse_args()
    
    occupy_gpu_memory(args.gpu, args.mb, args.hold)