import torch
import torch.nn as nn
import math
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from thop import profile


class Patch_Embedding(nn.Module):
    """
    Patch_Embedding模块：将输入图像分割为多个小块，并将每个小块映射为一个高维嵌入向量。
    """
    def __init__(self, Image=128, Patch=4, Input_C=2, Embedding_Dim_C=160, Norm_L=None):
        """
        初始化PatchEmbedding模块。

        参数：
        - Image: int，输入图像的大小（假设为方形图像，默认128）。
        - Patch: int，每个patch的大小（假设为正方形，默认4）。
        - Input_C: int，输入图像的通道数（默认2）。
        - Embedding_Dim_C: int，每个patch的嵌入维度（默认160）。
        - Norm_L: 可选，嵌入后用于归一化的层。
        """
        super().__init__()

        # 确保Image和Patch的格式为元组，方便支持图像和patch大小
        Image = to_2tuple(Image)
        Patch = to_2tuple(Patch)

        # 计算图像分割为patch后的网格大小（行数和列数）
        Patches_rs = [Image[0] // Patch[0], Image[1] // Patch[1]]

        # 保存图像和patch相关的属性
        self.Image = Image
        self.Patch = Patch
        self.Patches_rs = Patches_rs

        # 总的patch数量
        self.All_Patches = Patches_rs[0] * Patches_rs[1]

        # 保存输入通道数和目标嵌入维度
        self.Input_C = Input_C
        self.Embedding_Dim_C = Embedding_Dim_C

        # 如果指定了归一化层，则进行初始化，否则设置为None
        self.Norm = Norm_L(Embedding_Dim_C) if Norm_L else None

    def forward(self, x):
        # 将输入张量展平为 (B, C, H*W)
        x = x.flatten(2)

        # 转置后形状为 (B, H*W, C)
        x = x.transpose(1, 2)

        # 如果有归一化层，则对每个patch的嵌入向量进行归一化
        if self.Norm is not None:
            x = self.Norm(x)

        return x


class Patch_UnEmbedding(nn.Module):
    """
        Patch_Unembedding模块：从高维的patch嵌入向量中恢复图像的空间分布。
    """
    def __init__(self, Image=128, Patch=4, Input_C=2, Embedding_Dim_C=160, Norm_L=None):
        """
        初始化Patch_UnEmbedding模块。

        参数：
        - Image: int，输入图像的大小（假设为方形图像，默认128）。
        - Patch: int，每个patch的大小（假设为正方形，默认4）。
        - Input_C: int，输入图像的通道数（默认3）。
        - Embedding_Dim_C: int，每个patch的嵌入维度（默认96）。
        - Norm_L: 可选，嵌入后用于归一化的层。
        """
        super().__init__()

        # 确保Image和Patch的格式为元组，方便支持图像和patch大小
        Image = to_2tuple(Image)
        Patch = to_2tuple(Patch)

        # 计算图像分割为patch后的网格大小（行数和列数）
        Patches_rs = [Image[0] // Patch[0], Image[1] // Patch[1]]

        # 保存图像和patch相关的属性
        self.Image = Image
        self.Patch = Patch
        self.Patches_rs = Patches_rs

        # 总的patch数量
        self.All_Patches = Patches_rs[0] * Patches_rs[1]

        # 保存输入通道数和目标嵌入维度
        self.Input_C = Input_C
        self.Embedding_Dim_C = Embedding_Dim_C

        # 如果指定了归一化层，则进行初始化，否则设置为None
        self.Norm = Norm_L(Embedding_Dim_C) if Norm_L else None

    def forward(self, x, x_size):
        # 获取输入张量的形状
        B, HW, C = x.shape

        # 将输入张量转置并恢复为图像的空间结构
        x = x.transpose(1, 2)
        x = x.view(B, self.Embedding_Dim_C, x_size[0], x_size[1])

        return x



#上采样
class Upsample(nn.Sequential):
    def __init__(self, features):
        super(Upsample, self).__init__()
        # 上采样模块：通过卷积操作增加特征通道数，再使用PixelShuffle进行空间上采样。
        self.Conv2d_1 = nn.Conv2d(in_channels=features, out_channels=4 * features,
                                  kernel_size=3, stride=1, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)

    def forward(self, x):
        x = self.Conv2d_1(x)
        x = self.pixel_shuffle(x)
        return x



def C_Huafen(x, C_dim):
    """
    将输入张量划分为多个非重叠的小窗口。
    """

    # 获取输入张量的形状
    B, H, W, C = x.shape

    # 将张量重塑为包含窗口的形状
    x = x.view(B, H // C_dim, C_dim, W // C_dim, C_dim, C)

    # 调整张量顺序以便窗口连续排列
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()

    # 将张量展平为窗口形状
    windows = x.view(-1, C_dim, C_dim, C)

    return windows

def C_Huifu(C_window, C_dim, H, W):
    """
    从划分的窗口重建原始图像张量。
    """

    # 计算批大小 B
    B = int(C_window.shape[0] / (H * W / C_dim / C_dim))

    # 将窗口张量重塑为原始张量的分块形状
    x = C_window.view(B, H // C_dim, W // C_dim, C_dim, C_dim, -1)

    # 调整张量顺序以便恢复原始结构
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()

    # 将张量展平为原始图像形状：(B, H, W, C)
    x = x.view(B, H, W, -1)

    return x




class C_Attention(nn.Module):
    """
    基于窗口的多头自注意力（MHSA）机制, 在非重叠的窗口内计算自注意力
    """
    def __init__(self, dim, C_dim, heads, qkv_bias=True, qk_scale=None, drop=0):
        """
        初始化窗口多头自注意力模块。

        参数：
        - dim: int，输入特征的维度。
        - C_dim: tuple，窗口的尺寸 (height, width)。
        - heads: int，多头注意力的头数。
        - qkv_bias: bool，是否为 QKV 添加偏置。
        - qk_scale: 缩放因子，用于缩放 QK 点积（默认为头维度的平方根倒数）。
        - drop: 注意力权重和输出的丢弃概率。
        """
        super().__init__()
        self.dim = dim
        self.C_dim = C_dim
        self.heads = heads

        # 每个注意力头的维度
        head_dim = dim // heads
        # 使用自定义或默认的缩放因子
        self.scale = qk_scale or head_dim ** -0.5

        # 定义相对位置偏置表，用于建模窗口内的位置信息
        self.Xiangduiweizhi_Biaoge = nn.Parameter(
            torch.zeros((2 * C_dim[0] - 1) * (2 * C_dim[1] - 1), heads)
        )

        # 定义 Q、K、V 的线性映射
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.drop = nn.Dropout(drop)
        self.proj = nn.Linear(dim, dim)

        # 计算相对位置的索引
        Xd_high = torch.arange(self.C_dim[0])                # 水平方向的坐标
        Xd_width = torch.arange(self.C_dim[1])                # 垂直方向的坐标
        Xd = torch.stack(torch.meshgrid([Xd_high, Xd_width]))  # 创建网格坐标
        Xd_flatten = torch.flatten(Xd, 1)

        # 计算相对坐标差值
        Rl_Xd = Xd_flatten[:, :, None] - Xd_flatten[:, None, :]
        Rl_Xd = Rl_Xd.permute(1, 2, 0).contiguous()

        # 偏移坐标以避免负索引
        Rl_Xd[:, :, 0] += self.C_dim[0] - 1
        Rl_Xd[:, :, 1] += self.C_dim[1] - 1

        # 将水平方向索引乘以宽度扩展
        Rl_Xd[:, :, 0] *= 2 * self.C_dim[1] - 1

        # 计算最终的相对位置索引
        Rl_ind = Rl_Xd.sum(-1)

        # 注册为不可训练的缓冲区
        self.register_buffer("Rl_ind", Rl_ind)

        # 初始化相对位置偏置表，采用截断正态分布
        trunc_normal_(self.Xiangduiweizhi_Biaoge, std=.02)

        # 定义 softmax 用于计算注意力分数
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):

        # 获取输入的形状
        B_, N, C = x.shape

        # 计算 Q, K, V，并调整维度
        qkv = self.qkv(x).reshape(B_, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 缩放 Q
        q = q * self.scale

        # 计算 QK 点积，得到注意力分数
        attn = (q @ k.transpose(-2, -1))

        # 加入相对位置偏置
        Rl_Pbias = self.Xiangduiweizhi_Biaoge[self.Rl_ind.view(-1)].view(
            self.C_dim[0] * self.C_dim[1], self.C_dim[0] * self.C_dim[1], -1)
        Rl_Pbias = Rl_Pbias.permute(2, 0, 1).contiguous()
        attn = attn + Rl_Pbias.unsqueeze(0)

        # 如果提供了掩码，应用到注意力分数中
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.heads, N, N)

        # 通过 softmax 归一化注意力分数
        attn = self.softmax(attn)
        attn = self.drop(attn)

        # 根据注意力分数加权 V
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.drop(x)
        return x




class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)

        # 激活函数 (default: GELU)
        self.act = act_layer()

        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x





#
class SwinTransformerBlock(nn.Module):
    """
     Swin Transformer Block, 包含了移位窗口注意力和前馈 MLP 层.
    """
    def __init__(self, dim, input_resolution, num_heads, window_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        # 确保移位大小与窗口大小的合理性
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = C_Attention(
            dim, C_dim=to_2tuple(self.window_size), heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # 如果 shift_size > 0, 使用掩码
        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        """
        创建一个掩码用于移位窗口注意力，避免窗口之间的重叠。
        """
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))

        # 定义窗口划分的范围：水平和垂直方向切片
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))

        cnt = 0
        # 为每个切片区域赋值计数器
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        # 对图像掩码进行窗口划分
        mask_windows = C_Huafen(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape

        shortcut = x
        x = x.view(B, H, W, C)

        # 如果 shift_size > 0，进行移位操作
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # 将输入图像进行窗口划分
        x_windows = C_Huafen(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # 计算注意力，如果输入分辨率与当前尺寸一致，则使用已有的掩码，否则重新计算
        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)
        else:
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))

        # 将注意力结果重塑回窗口形状
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = C_Huifu(attn_windows, self.window_size, H, W)

        # 如果 shift_size > 0，恢复移位操作
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)  # 重塑回原形状

        #残差连接
        x = shortcut + self.drop_path(x)

        #归一化、MLP和残差连接
        x1 = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x + x1

        return x






class BasicLayer(nn.Module):
    """
    Swin Transformer 的基础层模块，由多个 Swin Transformer Blocks 和下采样组成。
    """
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        #初始化模块
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim,
                                 input_resolution=input_resolution,
                                 num_heads=num_heads,
                                 window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias,
                                 qk_scale=qk_scale,
                                 drop=drop,
                                 attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer) if downsample else None

    def forward(self, x, x_size):
        for blk in self.blocks:
            x = checkpoint.checkpoint(blk, x, x_size) if self.use_checkpoint else blk(x, x_size)

        if self.downsample is not None:
            x = self.downsample(x)
        return x



#Residual Groups(RG).
class RG(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 img_size=128, patch_size=1, resi_connection='1conv'):
        super(RG, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = BasicLayer(dim=dim,
                                         input_resolution=input_resolution,
                                         depth=depth,
                                         num_heads=num_heads,
                                         window_size=window_size,
                                         mlp_ratio=mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop, attn_drop=attn_drop,
                                         drop_path=drop_path,
                                         norm_layer=norm_layer,
                                         downsample=downsample,
                                         use_checkpoint=use_checkpoint)

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

        self.patch_embed = Patch_Embedding(
            Image=img_size, Patch=patch_size, Input_C=1, Embedding_Dim_C=dim,
            Norm_L=None)

        self.patch_unembed = Patch_UnEmbedding(
            Image=img_size, Patch=patch_size, Input_C=1, Embedding_Dim_C=dim,
            Norm_L=None)

    def forward(self, x, x_size):
        x1 = self.residual_group(x, x_size)
        x2 = self.patch_unembed(x1, x_size)
        x3 = self.conv(x2)
        x4 = self.patch_embed(x3) + x
        return x4


class SIST(nn.Module):
    """
      SIST (Seismic Image Super-Resolution Transformer)
    """
    def __init__(self, img_size=128, patch_size=1, in_chans=2,
                 embed_dim=160, depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6],
                 window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, upscale=2, img_range=1.,
                 **kwargs):
        super(SIST, self).__init__()

        self.img_range = img_range
        self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.window_size = window_size

        # 第一层卷积，用于输入图像处理
        self.conv_first = nn.Conv2d(2, embed_dim, 3, 1, 1)

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        # Patch Embedding：图像到嵌入空间的转换
        self.patch_embed = Patch_Embedding(
            Image=img_size, Patch=patch_size, Input_C=embed_dim, Embedding_Dim_C=embed_dim,
            Norm_L=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.All_Patches
        patches_rs = self.patch_embed.Patches_rs
        self.patches_rs = patches_rs

        # Patch Unembedding：将嵌入特征转换回图像空间
        self.patch_unembed = Patch_UnEmbedding(
            Image=img_size, Patch=patch_size, Input_C=embed_dim, Embedding_Dim_C=embed_dim,
            Norm_L=norm_layer if self.patch_norm else None)


        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # 多个 RG 层
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RG(dim=embed_dim,
                         input_resolution=(patches_rs[0],patches_rs[1]),depth=depths[i_layer],
                         num_heads=num_heads[i_layer],window_size=window_size,mlp_ratio=self.mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                         norm_layer=norm_layer,downsample=None,use_checkpoint=use_checkpoint,img_size=img_size,
                         patch_size=patch_size
                         )
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, 64, 3, 1, 1),
                                                      nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.upsample = Upsample(64)
        self.conv_last = nn.Conv2d(64, 1, 3, 1, 1)


    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x, x_size)

        x = self.norm(x)
        x = self.patch_unembed(x, x_size)

        return x

    def forward(self, x):
        x1 = self.conv_first(x)
        x2 = self.conv_first(x)
        x1 = self.conv_after_body(self.forward_features(x1))
        x1 = self.conv_before_upsample(x1)
        x1 = self.conv_last(self.upsample(x1))
        x2 = self.conv_before_upsample(x2)
        x2 = self.conv_last(self.upsample(x2))
        x = x1 + x2

        return x



if __name__ == '__main__':
    upscale = 2
    window_size = 8
    model = SIST(upscale=2, img_size=(128, 128),
                   window_size=window_size, img_range=1., in_chans=2,depths=[4, 4, 4, 4],
                   embed_dim=160, num_heads=[4, 4, 4, 4], mlp_ratio=4,act_cfg=dict(type='GELU'))
    #print(model)


    x = torch.randn((1, 2, 128, 128))

    #flops, params = profile(model, (x,))
    #print('flops: ', flops, 'params: ', params)

    x = model(x)
    print(x.shape)
