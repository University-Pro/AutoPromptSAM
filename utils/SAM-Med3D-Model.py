"""
调用完整的SAM-Med3D模型结构
"""
import medim
import torch

if __name__ == '__main__':
    model = medim.create_model("SAM-Med3D",
                               pretrained=True,
                               checkpoint_path="sam2_configs/sam_med3d_turbo.pth")
    
    image_encoer = model.image_encoder
    print(f'image encoder is {image_encoer}')

    x = torch.randn(1, 1, 128, 128, 128)
    
    print(f'input shape is {x.shape}')
    output = model(x)
    print(f'output shape is {output.shape}')

    print(f'model is {model}')