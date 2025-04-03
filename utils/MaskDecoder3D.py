"""
来自于SAM-Med3D.py的相关代码
能够处理3D张量
"""
import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Tuple, Type
from torch import Tensor, nn
import math
import numpy as np

class MLPBlock3D(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))

# 测试MLPBlock3D
def test_MLPBlock3D():
    # 定义超参数
    embedding_dim = 512
    mlp_dim = 2048

    # 初始化MLPBlock3D实例
    mlp_block = MLPBlock3D(embedding_dim=embedding_dim, mlp_dim=mlp_dim)

    # 创建输入张量，形状为(批次大小, 序列长度, 嵌入维度)
    batch_size = 32
    seq_len = 64
    x = torch.randn(batch_size, seq_len, embedding_dim)

    # 执行前向传播
    output = mlp_block(x)

    # 打印输入和输出的形状以验证是否正确
    print("Input shape:", x.shape)          # 应该是 (32, 64, 512)
    print("Output shape:", output.shape)    # 应该是 (32, 64, 512)

    # 另外，可以检查中间步骤的形状是否正确
    # 第一个线性层输出
    after_lin1 = mlp_block.lin1(x)
    print("Shape after lin1:", after_lin1.shape)  # 应该是 (32, 64, 2048)

    # 激活函数后
    after_act = mlp_block.act(after_lin1)
    print("Shape after activation:", after_act.shape)  # 同样为 (32, 64, 2048)

    # 第二个线性层输出，恢复到原嵌入维度
    after_lin2 = mlp_block.lin2(after_act)
    print("Shape after lin2:", after_lin2.shape)  # 应该是 (32, 64, 512)

    # 验证形状是否符合预期
    assert x.size() == output.size(), "输入和输出的形状不一致"

class TwoWayTransformer3D(nn.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
    ) -> None:
        """
        A transformer decoder that attends to an input image using
        queries whose positional embedding is supplied.

        Args:
          depth (int): number of layers in the transformer
          embedding_dim (int): the channel dimension for the input embeddings
          num_heads (int): the number of heads for multihead attention. Must
            divide embedding_dim
          mlp_dim (int): the channel dimension internal to the MLP block
          activation (nn.Module): the activation to use in the MLP block
        """
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()

        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock3D(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                )
            )

        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,
        point_embedding: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
          image_embedding (torch.Tensor): image to attend to. Should be shape
            B x embedding_dim x h x w for any h and w.
          image_pe (torch.Tensor): the positional encoding to add to the image. Must
            have the same shape as image_embedding.
          point_embedding (torch.Tensor): the embedding to add to the query points.
            Must have shape B x N_points x embedding_dim for any N_points.

        Returns:
          torch.Tensor: the processed point_embedding
          torch.Tensor: the processed image_embedding
        """
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        bs, c, x, y, z = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        # Prepare queries
        queries = point_embedding
        keys = image_embedding

        # Apply transformer blocks and final layernorm
        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )

        # Apply the final attention layer from the points to the image
        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys

class TwoWayAttentionBlock3D(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:
        """
        A transformer block with four layers: (1) self-attention of sparse
        inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
        block on sparse inputs, and (4) cross attention of dense inputs to sparse
        inputs.

        Arguments:
          embedding_dim (int): the channel dimension of the embeddings
          num_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
          activation (nn.Module): the activation of the mlp block
          skip_first_layer_pe (bool): skip the PE on the first layer
        """
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLPBlock3D(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )

        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(
        self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # Self attention block
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)

        # Cross attention block, tokens attending to image embedding
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys

def test_TwoWayAttentionBlock3D():
    # 初始化参数
    batch_size = 2
    num_tokens = 10  # 序列长度（如点提示数量）
    embedding_dim = 256
    num_heads = 8
    mlp_dim = 512

    # 初始化模块
    attention_block = TwoWayAttentionBlock3D(
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        skip_first_layer_pe=False
    )

    # 生成模拟输入 (B, N_tokens, C)
    queries = torch.randn(batch_size, num_tokens, embedding_dim)
    keys = torch.randn(batch_size, num_tokens, embedding_dim)
    query_pe = torch.randn(batch_size, num_tokens, embedding_dim)
    key_pe = torch.randn(batch_size, num_tokens, embedding_dim)

    # 执行前向传播
    new_queries, new_keys = attention_block(
        queries=queries,
        keys=keys,
        query_pe=query_pe,
        key_pe=key_pe
    )

    # 打印输入输出形状
    print("[TwoWayAttentionBlock3D] Input queries shape:", queries.shape)  # (2, 10, 256)
    print("[TwoWayAttentionBlock3D] Output queries shape:", new_queries.shape)  # (2, 10, 256)
    print("[TwoWayAttentionBlock3D] Output keys shape:", new_keys.shape)    # (2, 10, 256)

    # 验证形状一致性
    assert new_queries.shape == queries.shape, "Queries形状变化错误"
    assert new_keys.shape == keys.shape, "Keys形状变化错误"

    # 检查中间层维度
    # 1. 自注意力后的LayerNorm
    after_self_attn = attention_block.norm1(queries + queries)  # 模拟自注意力残差连接
    print("[TwoWayAttentionBlock3D] 自注意力+LayerNorm后形状:", after_self_attn.shape)  # (2, 10, 256)

    # 2. MLP块输出形状
    mlp_output = attention_block.mlp(new_queries)
    print("[TwoWayAttentionBlock3D] MLP输出形状:", mlp_output.shape)  # (2, 10, 256)

    print("TwoWayAttentionBlock3D测试通过！")

class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out

def test_Attention():
    # 初始化参数
    batch_size = 2
    num_tokens = 10  # 序列长度
    embedding_dim = 256
    num_heads = 8
    downsample_rate = 2  # 内部维度为128

    # 初始化Attention模块
    attention = Attention(
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        downsample_rate=downsample_rate
    )

    # 生成随机输入 Q/K/V (B, N_tokens, C)
    q = torch.randn(batch_size, num_tokens, embedding_dim)
    k = torch.randn(batch_size, num_tokens, embedding_dim)
    v = torch.randn(batch_size, num_tokens, embedding_dim)

    # 执行前向传播
    output = attention(q=q, k=k, v=v)

    # 打印输入输出形状
    print("[Attention] Input shape (Q/K/V):", q.shape)  # (2, 10, 256)
    print("[Attention] Output shape:", output.shape)    # (2, 10, 256)

    # 验证输出形状与输入Q一致
    assert output.shape == q.shape, "Attention输入输出形状不一致"

    # 验证分头逻辑
    internal_dim = embedding_dim // downsample_rate  # 128
    assert internal_dim % num_heads == 0, "num_heads必须能整除internal_dim"

    # 检查内部投影维度
    q_proj = attention.q_proj(q)  # (2, 10, 128)
    print("[Attention] Q投影后形状:", q_proj.shape)
    assert q_proj.shape[-1] == internal_dim

    # 检查分头后的形状
    q_separated = attention._separate_heads(q_proj, num_heads)  # (2, 8, 10, 16)
    print("[Attention] 分头后形状:", q_separated.shape)
    assert q_separated.shape == (batch_size, num_heads, num_tokens, internal_dim // num_heads)

    print("Attention测试通过！")

class LayerNorm3d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
        return x

def test_LayerNorm3D():
    # 初始化参数
    batch_size = 2
    num_channels = 64
    D, H, W = 4, 4, 4  # 3D体积的尺寸

    # 初始化LayerNorm3D实例
    layer_norm = LayerNorm3d(num_channels=num_channels)

    # 生成随机输入数据 (B, C, D, H, W)
    x = torch.randn(batch_size, num_channels, D, H, W)

    # 执行前向传播
    output = layer_norm(x)

    # 打印输入输出形状
    print("[LayerNorm3D] Input shape:", x.shape)    # (2, 64, 4, 4, 4)
    print("[LayerNorm3D] Output shape:", output.shape) # (2, 64, 4, 4, 4)

    # 验证形状一致性
    assert output.shape == x.shape, "LayerNorm3D输入输出形状不一致"

    # 验证归一化效果：每个通道的均值接近0，方差接近1
    eps = 1e-6
    output_np = output.detach().numpy()
    channel_mean = np.mean(output_np, axis=(2, 3, 4))  # 沿空间维度计算均值
    channel_var = np.var(output_np, axis=(2, 3, 4))    # 沿空间维度计算方差

    print("[LayerNorm3D] 通道均值范围:", np.abs(channel_mean).max())  # 应接近0
    print("[LayerNorm3D] 通道方差范围:", np.abs(channel_var - 1.0).max())  # 应接近0

    assert np.all(np.abs(channel_mean) < eps), "归一化后均值偏离0"
    assert np.all(np.abs(channel_var - 1.0) < eps), "归一化后方差偏离1"

    print("LayerNorm3D测试通过！")

class MaskDecoder3D(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        # transformer: nn.Module ,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        # self.transformer = transformer
        self.transformer = TwoWayTransformer3D(
                depth=2,
                embedding_dim=self.transformer_dim,
                mlp_dim=2048,
                num_heads=8,
            )

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose3d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm3d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose3d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        # Select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]

        # Prepare output
        return masks, iou_pred

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        if image_embeddings.shape[0] != tokens.shape[0]:
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        else:
            src = image_embeddings
        src = src + dense_prompt_embeddings
        if image_pe.shape[0] != tokens.shape[0]:
            pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        else:
            pos_src = image_pe
        b, c, x, y, z = src.shape

        # Run the transformer
        # import IPython; IPython.embed()
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, x, y, z)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, x, y, z = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, x * y * z)).view(b, -1, x, y, z)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred

# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x

def test_maskdecoder():
    # 初始化参数
    batch_size = 2
    transformer_dim = 256
    D, H, W = 4, 4, 4  # 3D体积的尺寸
    num_points = 5       # 稀疏提示的点数
    num_mask_tokens = 4  # num_multimask_outputs + 1 (3+1)

    # 创建MaskDecoder3D实例
    mask_decoder = MaskDecoder3D(
        transformer_dim=transformer_dim,
        num_multimask_outputs=3,
        activation=nn.GELU,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
    )

    # 生成模拟输入数据
    image_embeddings = torch.randn(batch_size, transformer_dim, D, H, W)
    image_pe = torch.randn(batch_size, transformer_dim, D, H, W)
    sparse_prompt_embeddings = torch.randn(batch_size, num_points, transformer_dim)
    dense_prompt_embeddings = torch.randn(batch_size, transformer_dim, D, H, W)

    # 执行前向传播
    masks, iou_pred = mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=image_pe,
        sparse_prompt_embeddings=sparse_prompt_embeddings,
        dense_prompt_embeddings=dense_prompt_embeddings,
        multimask_output=True,
    )

    # 打印输入输出形状
    print("Image embeddings shape:", image_embeddings.shape)
    print("Sparse prompt embeddings shape:", sparse_prompt_embeddings.shape)
    print("Dense prompt embeddings shape:", dense_prompt_embeddings.shape)
    print("Output masks shape:", masks.shape)
    print("IoU predictions shape:", iou_pred.shape)

    # 验证输出形状
    expected_mask_shape = (batch_size, 3, 16, 16, 16)  # 两次上采样4->8->16
    expected_iou_shape = (batch_size, 3)

    assert masks.shape == expected_mask_shape, f"Expected masks shape {expected_mask_shape}, got {masks.shape}"
    assert iou_pred.shape == expected_iou_shape, f"Expected IoU shape {expected_iou_shape}, got {iou_pred.shape}"

    print("All test cases passed!")

if __name__=="__main__":
    test_LayerNorm3D()
    test_Attention()
    test_TwoWayAttentionBlock3D()
    test_maskdecoder()  # 之前的MaskDecoder测试