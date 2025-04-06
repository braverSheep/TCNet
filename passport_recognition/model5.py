import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
# from thop import profile
from torch.cuda.amp import autocast
from torchvision import models
from torchvision.models import ResNet18_Weights
import math

# class FocalLoss(nn.Module):
#     def __init__(self, weight=None, reduction='mean', gamma=0, eps=1e-7):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         self.eps = eps
#         self.ce = torch.nn.CrossEntropyLoss(weight=weight, reduction=reduction)

#     def forward(self, input, target):
#         logp = self.ce(input, target)
#         p = torch.exp(-logp)
#         loss = (1 - p) ** self.gamma * logp
#         return loss.mean()

class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        is_avg_pool = True,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool:  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (int or None): Input resolution for calculating the relative positional
                parameter size.
        """
        super().__init__()
        self.is_avg_pool = is_avg_pool
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim_out)

        self.use_rel_pos = use_rel_pos
        self.down_scale = nn.Conv2d(dim_out,dim_out,2,2,0)
        # print('dim',dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0,2,3,1)
        # print('x.shape',x.shape)
        B, H, W, _ = x.shape # [8, 128, 32, 32]
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)
        x = x.permute(0,3,1,2) # Return shape [B, C, H, W]
        if self.is_avg_pool:
            x = F.avg_pool2d(x, kernel_size=2)
            # x = self.down_scale(x)

        return x

class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        kernel_size = [8, 8],
        stride = [8, 8],
        padding = [0, 0],
        in_chans = 3,
        embed_dim = 768,
    ) -> None:
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int):  embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        return x

class DynamicWeightingModule(nn.Module):
    def __init__(self, input_channels):
        super(DynamicWeightingModule, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  
        self.weight_generator = nn.Sequential(
            nn.Linear(input_channels * 2, 128),  
            nn.ReLU(),
            nn.Linear(128, 2),  
            nn.Softmax(dim=1)  
        )
        # self.fusing = nn.Conv2d(in_channels=input_channels * 2,out_channels=input_channels,stride=1,kernel_size=1,padding=0)

    def forward(self, local_feature, global_feature):
       
        local_feature_pooled = self.global_avg_pool(local_feature).view(local_feature.size(0), -1)  # [B, C]
        global_feature_pooled = self.global_avg_pool(global_feature).view(global_feature.size(0), -1)  # [B, C]

        
        combined_feature = torch.cat([local_feature_pooled, global_feature_pooled], dim=1)  # [B, 2*C]

        
        weights = self.weight_generator(combined_feature) 
       
        local_weight, global_weight = weights[:, 0].unsqueeze(1).unsqueeze(2).unsqueeze(3), weights[:, 1].unsqueeze(1).unsqueeze(2).unsqueeze(3)  # [B, 1, 1, 1]

        
        local_feature = local_weight * local_feature
        global_feature = global_weight * global_feature
        # print('local_feature.shape',local_feature.shape)
        # print('global_feature.shape',global_feature.shape)
        fused_feature = local_weight * local_feature + global_weight * global_feature  # [B, C, H, W]
        # cat = torch.cat((local_feature, global_feature), dim=1) 
        # fused_feature = self.fusing(cat)
        return fused_feature



class CNN_Tran(nn.Module):
    def __init__(self, img_size,patch_size,embed_dim,num_class):
        super(CNN_Tran,self).__init__()

        self.conv = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,stride=2,padding=1)
        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=3,
            embed_dim=embed_dim
        )
        
        self.pos_encoding = nn.Parameter(torch.randn(1, 64, 64, 64))

        pretrained = True 
        resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.layers = nn.Sequential(*list(resnet.children()))
        self.loc_block1 = self.layers[4]#64，64
        self.loc_block2 = self.layers[5]#64，128
        self.loc_block3 = self.layers[6]#128,256
        self.loc_block4 = self.layers[7]#256,512
        
        self.glo_blcok1 = Attention(dim=64,dim_out=64, num_heads=4, is_avg_pool=False)
        self.glo_blcok2 = Attention(dim=64,dim_out=128, num_heads=4)
        self.glo_blcok3 = Attention(dim=128,dim_out=256, num_heads=4)
        self.glo_blcok4 = Attention(dim=256,dim_out=512, num_heads=4)
        
        self.fus_block1 = DynamicWeightingModule(input_channels=64)
        self.fus_block2 = DynamicWeightingModule(input_channels=128)
        self.fus_block3 = DynamicWeightingModule(input_channels=256)
        self.fus_block4 = DynamicWeightingModule(input_channels=512)
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.chassifier = nn.Linear(in_features=512, out_features=num_class, bias=True)

    def forward(self, x):
        x = x.type(torch.float)
        # block1
        cnn_out = self.conv(x)
        #patch_out = self.patch_embed(x) + positional_encoding_2d(x.shape[0], 64, 64, 64).to('cuda')
        patch_out = self.patch_embed(x) + self.pos_encoding.expand(x.shape[0], -1, -1, -1)

        loc_out = self.loc_block1(cnn_out)
        glo_out = self.glo_blcok1(patch_out)
        # print(loc_out.shape, glo_out.shape)
        out_fuse = self.fus_block1(loc_out, glo_out)
        # for i in range(49):
        #     y = self.global_avg_pool(out_fuse)
        # print(out_fuse.shape)

        # block2
        cnn_out = out_fuse
        patch_out = out_fuse
        loc_out = self.loc_block2(cnn_out)
        glo_out = self.glo_blcok2(patch_out)
        out_fuse = self.fus_block2(loc_out, glo_out)
        # print("out_fuse",out_fuse.shape)
        # print(out_fuse.shape)

        # block3
        cnn_out = out_fuse
        patch_out = out_fuse
        loc_out = self.loc_block3(cnn_out)
        glo_out = self.glo_blcok3(patch_out)
        out_fuse = self.fus_block3(loc_out, glo_out)
        # print(out_fuse.shape)

        # block4
        cnn_out = out_fuse
        patch_out = out_fuse
        loc_out = self.loc_block4(cnn_out)
        glo_out = self.glo_blcok4(patch_out)
        out_fuse = self.fus_block4(loc_out, glo_out)
        # print(out_fuse.shape)

        feature_maps = out_fuse
       
        # out_fuse = out_fuse[:, 0:256,:,:]
        
        pool_features = self.global_avg_pool(out_fuse)
        pool_features = pool_features.view(pool_features.size(0), -1)
        clss_out = self.chassifier(pool_features)

        return clss_out #, feature_maps


if __name__ == '__main__':
    import time
    x = torch.randn(8, 3, 128, 128).cuda()
    # resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    start = time.time()
    net = CNN_Tran(img_size=128, patch_size=2, embed_dim=64, num_class=4).cuda()
    y = net(x)
    end = time.time()
    print('end-start=',end-start)
    print(net)
    # print(y.shape)
   
    # from ptflops import get_model_complexity_info

    
    # macs, params = get_model_complexity_info(net, (3, 128, 128), as_strings=True, print_per_layer_stat=True) # True，则输出的 FLOPs（MACs）和参数数量会以字符串格式返回

    # print(f"FLOPs: {macs}")
    
