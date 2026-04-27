import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ConvBlock(nn.Module):
    def __init__(self, channels):
        super(ConvBlock, self).__init__()
        self.dw = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels, bias=False)
        self.pw = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        return self.act(x)


class PatchEmbed(nn.Module):
    def __init__(self, in_chans=3, embed_dim=128, patch_size=4):
        super(PatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        return self.norm(x)


class PatchMerging(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(PatchMerging, self).__init__()
        self.reduction = nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2, bias=False)
        self.norm = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        x = self.reduction(x)
        return self.norm(x)


class VisualExtractor(nn.Module):
    def __init__(self, args):
        super(VisualExtractor, self).__init__()
        self.visual_mode = getattr(args, 'visual_mode', 'resnet')
        self.visual_extractor = args.visual_extractor
        self.pretrained = args.visual_extractor_pretrained
        self.cross_scan_fuse = getattr(args, 'cross_scan_fuse', 'concat')

        if self.visual_mode == 'resnet':
            model = getattr(models, self.visual_extractor)(pretrained=self.pretrained)
            modules = list(model.children())[:-2]
            self.model = nn.Sequential(*modules)
            self.avg_fnt = nn.AdaptiveAvgPool2d((1, 1))
        else:
            embed_dim = getattr(args, 'vis_embed_dim', 128)
            out_dim = args.d_vf
            patch_size = getattr(args, 'vis_patch_size', 4)

            self.patch_embed = PatchEmbed(in_chans=3, embed_dim=embed_dim, patch_size=patch_size)
            self.stage1 = nn.Sequential(ConvBlock(embed_dim), ConvBlock(embed_dim))
            self.merge1 = PatchMerging(embed_dim, embed_dim * 2)
            self.stage2 = nn.Sequential(ConvBlock(embed_dim * 2), ConvBlock(embed_dim * 2))
            self.merge2 = PatchMerging(embed_dim * 2, out_dim)
            self.stage3 = nn.Sequential(ConvBlock(out_dim), ConvBlock(out_dim))

            if self.cross_scan_fuse == 'concat':
                self.fuse = nn.Linear(out_dim * 4, out_dim)
            else:
                self.scan_weight = nn.Parameter(torch.zeros(4))

    def _flatten_row_major(self, feat_2d):
        return feat_2d.flatten(2).transpose(1, 2).contiguous()

    def _flatten_col_major(self, feat_2d):
        feat_2d = feat_2d.permute(0, 1, 3, 2).contiguous()
        return feat_2d.flatten(2).transpose(1, 2).contiguous()

    def _cross_scan_fuse(self, feat_2d):
        s_row = self._flatten_row_major(feat_2d)
        s_col = self._flatten_col_major(feat_2d)
        s_row_rev = torch.flip(s_row, dims=[1])
        s_col_rev = torch.flip(s_col, dims=[1])

        if self.cross_scan_fuse == 'concat':
            s = torch.cat([s_row, s_col, s_row_rev, s_col_rev], dim=-1)
            s = self.fuse(s)
        else:
            w = F.softmax(self.scan_weight, dim=0)
            s = w[0] * s_row + w[1] * s_col + w[2] * s_row_rev + w[3] * s_col_rev
        return s

    def forward(self, images):
        if self.visual_mode == 'resnet':
            patch_feats = self.model(images)
            avg_feats = self.avg_fnt(patch_feats).flatten(1)
            batch_size, feat_size, _, _ = patch_feats.shape
            patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1).contiguous()
            return patch_feats, avg_feats

        x = self.patch_embed(images)
        x = self.stage1(x)
        x = self.merge1(x)
        x = self.stage2(x)
        x = self.merge2(x)
        x = self.stage3(x)

        patch_feats = self._cross_scan_fuse(x)
        avg_feats = patch_feats.mean(dim=1)
        return patch_feats, avg_feats
