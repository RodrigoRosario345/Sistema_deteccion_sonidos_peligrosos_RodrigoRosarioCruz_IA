import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import repeat
import collections.abc
import math
import warnings

from torch.nn.init import _calculate_fan_in_and_fan_out


# from PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True, patch_stride = 16):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patch_stride = to_2tuple(patch_stride)
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.grid_size = (img_size[0] // patch_stride[0], img_size[1] // patch_stride[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        
        padding = ((patch_size[0] - patch_stride[0]) // 2, (patch_size[1] - patch_stride[1]) // 2)

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x

class Mlp(nn.Module):
    """ MLP como se utiliza en Vision Transformer, MLP-Mixer y redes relacionadas
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
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

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        # Calculamos la función de distribución acumulativa normal estándar
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Los valores se generan utilizando una distribución uniforme truncada y
        # luego usando la CDF inversa para la distribución normal.
        # Obtener los valores de CDF superior e inferior
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Llenar uniformemente el tensor con valores de [l, u], luego traducir a
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Usar la transformación de CDF inversa para la distribución normal para obtener
        # normal estándar truncada
        tensor.erfinv_()

        # Transformar a la media y desviación estándar adecuadas
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Ajustar para asegurar que está en el rango adecuado
        tensor.clamp_(min=a, max=b)
        return tensor



def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # tipo: (Tensor, float, float, float, float) -> Tensor
    r"""Llena el Tensor de entrada con valores extraídos de una
    distribución normal truncada. Los valores se extraen efectivamente de la
    distribución normal :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    con valores fuera de :math:`[a, b]` redibujados hasta que estén dentro
    de los límites. El método utilizado para generar los valores aleatorios
    funciona mejor cuando :math:`a \leq \text{mean} \leq b`.
    Argumentos:
        tensor: un `torch.Tensor` n-dimensional
        mean: la media de la distribución normal
        std: la desviación estándar de la distribución normal
        a: el valor mínimo de corte
        b: el valor máximo de corte
    Ejemplos:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """

    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='normal'):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == 'fan_in':
        denom = fan_in
    elif mode == 'fan_out':
        denom = fan_out
    elif mode == 'fan_avg':
        denom = (fan_in + fan_out) / 2

    variance = scale / denom

    if distribution == "truncated_normal":
        # constante es stddev de normal estándar truncada a (-2, 2)
        trunc_normal_(tensor, std=math.sqrt(variance) / .87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
    variance_scaling_(tensor, mode='fan_in', distribution='truncated_normal')
