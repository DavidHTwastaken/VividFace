import functools
from typing import Optional, Tuple, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from transformers import CLIPVisionModel, Dinov2Model

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=dilation,
                     groups=groups,
                     bias=False,
                     dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class IBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1):
        super(IBasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-05,)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05,)
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-05,)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out



class IResNet(nn.Module):
    fc_scale = 7 * 7
    def __init__(self,
                 block, layers, dropout=0, num_features=512, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, fp16=False):
        super(IResNet, self).__init__()
        self.extra_gflops = 0.0
        self.fp16 = fp16
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=1e-05)
        self.prelu = nn.PReLU(self.inplanes)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block,
                                       128,
                                       layers[1],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block,
                                       256,
                                       layers[2],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block,
                                       512,
                                       layers[3],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.bn2 = nn.BatchNorm2d(512 * block.expansion, eps=1e-05,)
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.fc = nn.Linear(512 * block.expansion * self.fc_scale, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-05)
        # nn.init.constant_(self.features.weight, 1.0)
        # self.features.weight.requires_grad = False


    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-05, ),
            )
        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.groups,
                  self.base_width, previous_dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes,
                      planes,
                      groups=self.groups,
                      base_width=self.base_width,
                      dilation=self.dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn2(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.features(x)
        return x


def _iresnet(arch, block, layers, pretrained, **kwargs):
    model = IResNet(block, layers, **kwargs)
    model.load_state_dict(torch.load(pretrained, map_location="cpu"), strict=True)
    model.eval()
    model.requires_grad_(False)
    return model


def iresnet18(pretrained=None, **kwargs):
    return _iresnet('iresnet18', IBasicBlock, [2, 2, 2, 2], pretrained, **kwargs)


def iresnet34(pretrained=None, **kwargs):
    return _iresnet('iresnet34', IBasicBlock, [3, 4, 6, 3], pretrained,  **kwargs)


def iresnet50(pretrained=None, **kwargs):
    return _iresnet('iresnet50', IBasicBlock, [3, 4, 14, 3], pretrained, **kwargs)


def iresnet100(pretrained=None, **kwargs):
    return _iresnet('iresnet100', IBasicBlock, [3, 13, 30, 3], pretrained, **kwargs)


def iresnet200(pretrained=None, **kwargs):
    return _iresnet('iresnet200', IBasicBlock, [6, 26, 60, 6], pretrained, **kwargs)



class Generator_Adain_Upsample(nn.Module):
    def __init__(self, input_nc, n_blocks=6, norm_layer=nn.BatchNorm2d):
        assert (n_blocks >= 0)
        super(Generator_Adain_Upsample, self).__init__()

        activation = nn.LeakyReLU(0.2)

        self.first_layer = nn.Sequential(
            nn.Conv2d(input_nc, 64, kernel_size=7, padding=3, padding_mode="reflect"),
            norm_layer(64),
            activation
        )

        self.down1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            norm_layer(128),
            activation
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            norm_layer(256),
            activation
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            norm_layer(512),
            activation
        )


    def forward(self, inputs):
        x = self.first_layer(inputs)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        return x


class FaceEmbedder(nn.Module):
    def __init__(
        self,
        arcface_path: 'IResNet100_WebFace42M.pth',
        dino_model_path: str = None
    ):
        super().__init__()
        print('Using New  with DoubleDino.')
        self.arcface_embed_dim = 512
        self.attr_embed_dim = 768
        self.arcface = iresnet100(arcface_path)
        self.arcface.eval()
        self.arcface.requires_grad_(False)

        self.dino = Dinov2Model.from_pretrained(dino_model_path)
        self.dino_embed_dim = self.dino.config.hidden_size

    @torch.no_grad()
    def get_id_feat(self, x) -> torch.Tensor:
        idemb = self.arcface(F.interpolate(x, size=(112, 112), mode='bicubic'))
        idemb = F.normalize(idemb, p=2, dim=-1)
        return idemb

    @torch.no_grad()
    def forward(self, face_pixel_values: torch.Tensor, mask: torch.Tensor = None, attr_pixel_values: torch.Tensor = None, dino_pixel_values: torch.Tensor = None):

        batch_size = face_pixel_values.size(0)
        if mask is None:
            mask = torch.ones(batch_size, device=face_pixel_values.device)

        validmask = mask > 0.5
        face_pixel_values = face_pixel_values[validmask]
        dino_pixel_values = dino_pixel_values[validmask]
        attr_pixel_values = attr_pixel_values[validmask]

        id_embed = self.arcface(face_pixel_values)
        id_embed = F.normalize(id_embed, p=2, dim=-1)
        full_id_embed = torch.zeros((batch_size, self.arcface_embed_dim), device=id_embed.device, dtype=id_embed.dtype)
        full_id_embed[validmask] = id_embed

        dino_input = TF.normalize((dino_pixel_values + 1.0)/2, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        dino_output = self.dino(dino_input, output_hidden_states=True)
        dino_embed = dino_output.hidden_states[-1][:, 1:, :]
        full_dino_embed = torch.zeros((batch_size, dino_embed.size(1), self.dino_embed_dim), device=dino_embed.device, dtype=dino_embed.dtype)
        full_dino_embed[validmask] = dino_embed

        attr_input = TF.normalize((attr_pixel_values + 1.0)/2, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        attr_output = self.dino(attr_input, output_hidden_states=True)
        attr_embed = attr_output.hidden_states[-1][:, 1:, :]
        full_attr_embed = torch.zeros((batch_size, attr_embed.size(1), self.attr_embed_dim), device=attr_embed.device, dtype=attr_embed.dtype)
        full_attr_embed[validmask] = attr_embed

        return {
            "id_embed": full_id_embed,
            "attr_embed": full_attr_embed,
            "dino_embed": full_dino_embed
        }



def get_similarity_transform_matrix(from_pts: torch.Tensor, to_pts: torch.Tensor) -> torch.Tensor:
    """
    Args:
        from_pts, to_pts: b x n x 2

    Returns:
        torch.Tensor: b x 3 x 3
    """
    mfrom = from_pts.mean(dim=1, keepdim=True)  # b x 1 x 2
    mto = to_pts.mean(dim=1, keepdim=True)  # b x 1 x 2

    a1 = (from_pts - mfrom).square().sum([1, 2], keepdim=False)  # b
    c1 = ((to_pts - mto) * (from_pts - mfrom)).sum([1, 2], keepdim=False)  # b

    to_delta = to_pts - mto
    from_delta = from_pts - mfrom
    c2 = (to_delta[:, :, 0] * from_delta[:, :, 1] - to_delta[:, :, 1] * from_delta[:, :, 0]).sum([1], keepdim=False)  # b

    a = c1 / a1
    b = c2 / a1
    dx = mto[:, 0, 0] - a * mfrom[:, 0, 0] - b * mfrom[:, 0, 1]  # b
    dy = mto[:, 0, 1] + b * mfrom[:, 0, 0] - a * mfrom[:, 0, 1]  # b

    ones_pl = torch.ones_like(a1)
    zeros_pl = torch.zeros_like(a1)

    return torch.stack([
        a, b, dx,
        -b, a, dy,
        zeros_pl, zeros_pl, ones_pl,
    ], dim=-1).reshape(-1, 3, 3)

def get_face_align_matrix(face_pts: torch.Tensor, target_pts: torch.Tensor):
    target_pts = target_pts.to(face_pts)
    if target_pts.dim() == 2:
        target_pts = target_pts.unsqueeze(0)
    if target_pts.size(0) == 1:
        target_pts = target_pts.broadcast_to(face_pts.shape)
    assert target_pts.shape == face_pts.shape
    return get_similarity_transform_matrix(face_pts, target_pts)

@functools.lru_cache(maxsize=128)
def _meshgrid(h, w) -> Tuple[torch.Tensor, torch.Tensor]:
    yy, xx = torch.meshgrid(
        torch.arange(h).float(),
        torch.arange(w).float(),
        indexing='ij'
    )
    return yy, xx

def inverted_warp_transform(coords: torch.Tensor, matrix: torch.Tensor):
    """ Inverted tanh-warp function.

    Args:
        coords (torch.Tensor): b x n x 2 (x, y). The transformed coordinates.
        matrix: b x 3 x 3. A matrix that transforms un-normalized coordinates 
            from the original image to the aligned yet not-warped image.
        warped_shape (tuple): [height, width].

    Returns:
        torch.Tensor: b x n x 2 (x, y). The original coordinates.
    """

    coords_homo = torch.cat([coords, torch.ones_like(coords[:, :, [0]])], dim=-1)  # b x n x 3

    inv_matrix = torch.linalg.inv(matrix)  # b x 3 x 3
    # inv_matrix = np.linalg.inv(matrix)
    coords_homo = torch.bmm(coords_homo, inv_matrix.permute(0, 2, 1))  # b x n x 3
    return coords_homo[:, :, :2] / coords_homo[:, :, [2, 2]]


def _forge_grid(
    matrix: torch.Tensor,
    output_shape: Tuple[int, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Forge transform maps with a given function `fn`.

    Args:
        output_shape (tuple): (b, h, w, ...).
        fn (Callable[[torch.Tensor], torch.Tensor]): The function that accepts 
            a bxnx2 array and outputs the transformed bxnx2 array. Both input 
            and output store (x, y) coordinates.

    Note: 
        both input and output arrays of `fn` should store (y, x) coordinates.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Two maps `X` and `Y`, where for each 
            pixel (y, x) or coordinate (x, y),
            `(X[y, x], Y[y, x]) = fn([x, y])`
    """
    batch_size = matrix.size(0)
    device = matrix.device
    h, w, *_ = output_shape
    yy, xx = _meshgrid(h, w)  # h x w
    yy = yy.unsqueeze(0).broadcast_to(batch_size, h, w).to(device)
    xx = xx.unsqueeze(0).broadcast_to(batch_size, h, w).to(device)

    in_xxyy = torch.stack([xx, yy], dim=-1).reshape([batch_size, h*w, 2])  # (h x w) x 2
    out_xxyy: torch.Tensor = inverted_warp_transform(in_xxyy, matrix)  # (h x w) x 2

    return out_xxyy.reshape(batch_size, h, w, 2)




def make_warp_grid(
    matrix: torch.Tensor,
    warped_shape: Tuple[int, int],
    orig_shape: Tuple[int, int]
):
    """
    Args:
        matrix: bx3x3 matrix.

        warped_shape: The target image shape to transform to.

    Returns:
        torch.Tensor: b x h x w x 2 (x, y).
    """
    orig_h, orig_w, *_ = orig_shape
    w_h = torch.tensor([orig_w, orig_h]).to(matrix).reshape(1, 1, 1, 2)
    grid = _forge_grid(matrix, warped_shape)
    grid = grid / w_h * 2 - 1
    return grid

# from torchvision.utils import make_grid, save_image

class IDLoss(nn.Module):
    def __init__(self, resnet_path="w600k_r50.pth", out_size=112):
        super().__init__()
        target_pts = np.array(
            [
                [38.2946, 51.6963],  # left eye
                [73.5318, 51.5014],  # right eye
                [56.0252, 71.7366],  # nose tip
                [41.5493, 92.3655],  # left mouth corner
                [70.7299, 92.2041],  # right mouth corner
            ],
        )
        old_size = 112
        target_pts = target_pts / old_size * out_size
        self.register_buffer("target_pts", torch.from_numpy(target_pts).float())
        self.iresnet = iresnet100(pretrained=resnet_path)

    @torch.no_grad()
    def similarity(self, images: torch.Tensor, kps: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        _, _, h, w = images.shape
        images = images.float()
        kps = kps * torch.Tensor([h, w]).to(images.device)

        matrix = get_face_align_matrix(kps, self.target_pts)
        grid = make_warp_grid(matrix, orig_shape=(h, w), warped_shape=(112, 112))
        faces = F.grid_sample(images, grid, mode="bilinear", align_corners=False)

        targets = F.interpolate(targets, size=(112, 112), mode="bilinear")
        target_emb = self.iresnet((targets - 0.5) / 0.5)
        face_emb = self.iresnet(faces)

        cosim = F.cosine_similarity(face_emb, target_emb, dim=-1)

        return cosim.mean().item()

    def forward(self, images: torch.Tensor, kps: torch.Tensor, targets: torch.Tensor, step) -> torch.Tensor:
        _, _, h, w = images.shape
        images = images.float()
        kps = kps * torch.Tensor([h, w]).to(images.device)
        if kps.sum() <= 0.01:
            faces = F.interpolate(images, size=(112, 112), mode="bilinear")
        else:
            matrix = get_face_align_matrix(kps, self.target_pts)
            grid = make_warp_grid(matrix, orig_shape=(h, w), warped_shape=(112, 112))
            faces = F.grid_sample(images, grid, mode="bilinear", align_corners=False)

        with torch.no_grad():
            targets = F.interpolate(targets, size=(112, 112), mode="bilinear")
            # save_image(make_grid(torch.cat([targets, (faces+1.0)/2.0]), nrow=4, padding = 4, normalize=False), f"sample_{step}_id.jpg")
            target_emb = self.iresnet(targets)
        face_emb = self.iresnet(faces)

        cosim = F.cosine_similarity(face_emb, target_emb, dim=-1)
        cosim = (1.0 - cosim).mean()

        return cosim

if __name__ == "__main__":

    model = FaceEmbedder(
        "IResNet100_WebFace42M.pth",
    ).cuda()
    x = torch.rand(4, 3, 112, 112).cuda()
    x = (x - 0.5) / 0.5
    y = torch.rand(4, 3, 224, 224).cuda()
    y = (y - 0.5) / 0.5
    z = model(x, None, y)
    print(z["id_embed"].shape, z["attr_embed"].shape)
