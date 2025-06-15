from dataclasses import dataclass
import math

import torch
import numpy as np
import random
import time
import trimesh
import torch.nn as nn
from einops import repeat, rearrange
from tqdm import trange
from itertools import product
from diffusers.models.modeling_utils import ModelMixin

import step1x3d_geometry
from step1x3d_geometry.utils.checkpoint import checkpoint
from step1x3d_geometry.utils.base import BaseModule
from step1x3d_geometry.utils.typing import *
from step1x3d_geometry.utils.misc import get_world_size, get_device

from .transformers.perceiver_1d import Perceiver
from .transformers.attention import ResidualCrossAttentionBlock
from .volume_decoders import HierarchicalVolumeDecoder, VanillaVolumeDecoder
from .surface_extractors import MCSurfaceExtractor, DMCSurfaceExtractor

from ..pipelines.pipeline_utils import smart_load_model
from safetensors.torch import load_file

VALID_EMBED_TYPES = ["identity", "fourier", "learned_fourier", "siren"]


class FourierEmbedder(nn.Module):
    def __init__(
        self,
        num_freqs: int = 6,
        logspace: bool = True,
        input_dim: int = 3,
        include_input: bool = True,
        include_pi: bool = True,
    ) -> None:
        super().__init__()

        if logspace:
            frequencies = 2.0 ** torch.arange(num_freqs, dtype=torch.float32)
        else:
            frequencies = torch.linspace(
                1.0, 2.0 ** (num_freqs - 1), num_freqs, dtype=torch.float32
            )

        if include_pi:
            frequencies *= torch.pi

        self.register_buffer("frequencies", frequencies, persistent=False)
        self.include_input = include_input
        self.num_freqs = num_freqs

        self.out_dim = self.get_dims(input_dim)

    def get_dims(self, input_dim):
        temp = 1 if self.include_input or self.num_freqs == 0 else 0
        out_dim = input_dim * (self.num_freqs * 2 + temp)

        return out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_freqs > 0:
            embed = (x[..., None].contiguous() * self.frequencies).view(
                *x.shape[:-1], -1
            )
            if self.include_input:
                return torch.cat((x, embed.sin(), embed.cos()), dim=-1)
            else:
                return torch.cat((embed.sin(), embed.cos()), dim=-1)
        else:
            return x


class LearnedFourierEmbedder(nn.Module):
    def __init__(self, input_dim, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        per_channel_dim = half_dim // input_dim
        self.weights = nn.Parameter(torch.randn(per_channel_dim))

        self.out_dim = self.get_dims(input_dim)

    def forward(self, x):
        # [b, t, c, 1] * [1, d] = [b, t, c, d] -> [b, t, c * d]
        freqs = (x[..., None] * self.weights[None] * 2 * np.pi).view(*x.shape[:-1], -1)
        fouriered = torch.cat((x, freqs.sin(), freqs.cos()), dim=-1)
        return fouriered

    def get_dims(self, input_dim):
        return input_dim * (self.weights.shape[0] * 2 + 1)


class Sine(nn.Module):
    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class Siren(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        w0=1.0,
        c=6.0,
        is_first=False,
        use_bias=True,
        activation=None,
        dropout=0.0,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.is_first = is_first

        weight = torch.zeros(out_dim, in_dim)
        bias = torch.zeros(out_dim) if use_bias else None
        self.init_(weight, bias, c=c, w0=w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None
        self.activation = Sine(w0) if activation is None else activation
        self.dropout = nn.Dropout(dropout)

    def init_(self, weight, bias, c, w0):
        dim = self.in_dim

        w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
        weight.uniform_(-w_std, w_std)

        if bias is not None:
            bias.uniform_(-w_std, w_std)

    def forward(self, x):
        out = F.linear(x, self.weight, self.bias)
        out = self.activation(out)
        out = self.dropout(out)
        return out


def get_embedder(embed_type="fourier", num_freqs=-1, input_dim=3, include_pi=True):
    if embed_type == "identity" or (embed_type == "fourier" and num_freqs == -1):
        return nn.Identity(), input_dim

    elif embed_type == "fourier":
        embedder_obj = FourierEmbedder(num_freqs=num_freqs, include_pi=include_pi)

    elif embed_type == "learned_fourier":
        embedder_obj = LearnedFourierEmbedder(in_channels=input_dim, dim=num_freqs)

    elif embed_type == "siren":
        embedder_obj = Siren(
            in_dim=input_dim, out_dim=num_freqs * input_dim * 2 + input_dim
        )

    else:
        raise ValueError(
            f"{embed_type} is not valid. Currently only supprts {VALID_EMBED_TYPES}"
        )
    return embedder_obj


###################### AutoEncoder
class DiagonalGaussianDistribution(ModelMixin, object):
    def __init__(
        self,
        parameters: Union[torch.Tensor, List[torch.Tensor]],
        deterministic=False,
        feat_dim=1,
    ):
        self.feat_dim = feat_dim
        self.parameters = parameters

        if isinstance(parameters, list):
            self.mean = parameters[0]
            self.logvar = parameters[1]
        else:
            self.mean, self.logvar = torch.chunk(parameters, 2, dim=feat_dim)

        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean)

    def sample(self):
        x = self.mean + self.std * torch.randn_like(self.mean)
        return x

    def kl(self, other=None, dims=(1, 2)):
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:
                return 0.5 * torch.mean(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar, dim=dims
                )
            else:
                return 0.5 * torch.mean(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=dims,
                )

    def nll(self, sample, dims=(1, 2)):
        if self.deterministic:
            return torch.Tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims,
        )

    def mode(self):
        return self.mean


class PerceiverCrossAttentionEncoder(ModelMixin, nn.Module):
    def __init__(
        self,
        use_downsample: bool,
        num_latents: int,
        embedder: FourierEmbedder,
        point_feats: int,
        embed_point_feats: bool,
        width: int,
        heads: int,
        layers: int,
        init_scale: float = 0.25,
        qkv_bias: bool = True,
        qk_norm: bool = True,
        use_ln_post: bool = False,
        use_flash: bool = False,
        use_checkpoint: bool = False,
        use_multi_reso: bool = False,
        resolutions: list = [],
        sampling_prob: list = [],
        with_sharp_data: bool = False,
    ):

        super().__init__()

        self.use_checkpoint = use_checkpoint
        self.num_latents = num_latents
        self.use_downsample = use_downsample
        self.embed_point_feats = embed_point_feats
        self.use_multi_reso = use_multi_reso
        self.resolutions = resolutions
        self.sampling_prob = sampling_prob

        if not self.use_downsample:
            self.query = nn.Parameter(torch.randn((num_latents, width)) * 0.02)

        self.embedder = embedder
        if self.embed_point_feats:
            self.input_proj = nn.Linear(self.embedder.out_dim * 2, width)
        else:
            self.input_proj = nn.Linear(self.embedder.out_dim + point_feats, width)

        self.cross_attn = ResidualCrossAttentionBlock(
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            use_flash=use_flash,
        )

        self.with_sharp_data = with_sharp_data
        if with_sharp_data:
            self.downsmaple_num_latents = num_latents // 2
            self.input_proj_sharp = nn.Linear(
                self.embedder.out_dim + point_feats, width
            )
            self.cross_attn_sharp = ResidualCrossAttentionBlock(
                width=width,
                heads=heads,
                init_scale=init_scale,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                use_flash=use_flash,
            )
        else:
            self.downsmaple_num_latents = num_latents

        self.self_attn = Perceiver(
            n_ctx=num_latents,
            width=width,
            layers=layers,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            use_flash=use_flash,
            use_checkpoint=use_checkpoint,
        )

        if use_ln_post:
            self.ln_post = nn.LayerNorm(width)
        else:
            self.ln_post = None

    def _forward(self, pc, feats, sharp_pc=None, sharp_feat=None):
        """

        Args:
            pc (torch.FloatTensor): [B, N, 3]
            feats (torch.FloatTensor or None): [B, N, C]

        Returns:

        """

        bs, N, D = pc.shape

        # Apply multi-resolution sampling FIRST (if enabled)
        if self.use_multi_reso:
            from torch_cluster import fps

            resolution = random.choice(self.resolutions, size=1, p=self.sampling_prob)[0]

            if resolution != N:
                flattened = pc.view(bs * N, D)  # bs*N, 64.      103,4096,3 -> 421888,3
                batch = torch.arange(bs).to(pc.device)  # 103
                batch = torch.repeat_interleave(batch, N)  # bs*N. 421888
                pos = flattened.to(torch.float16)
                ratio = 1.0 * resolution / N  # 0.0625
                idx = fps(pos, batch, ratio=ratio)  # 26368

                # Downsample all inputs
                pc = pc.view(bs * N, -1)[idx].view(bs, -1, D)

                if feats is not None:
                    feats = feats.view(bs * N, -1)[idx].view(bs, -1, feats.shape[-1])
                if sharp_pc is not None:
                    sharp_pc = sharp_pc.view(bs * N, -1)[idx].view(bs, -1, D)
                if sharp_feat is not None:
                    sharp_feat = sharp_feat.view(bs * N, -1)[idx].view(bs, -1, sharp_feat.shape[-1])

                bs, N, D = pc.shape

        # Continue with the original number of points
        # if no downsampling is applied or after multi-resolution sampling
        data = self.embedder(pc)
        if feats is not None:
            if self.embed_point_feats:
                feats = self.embedder(feats)
            data = torch.cat([data, feats], dim=-1)
        data = self.input_proj(data)

        if self.with_sharp_data:
            sharp_data = self.embedder(sharp_pc)
            if sharp_feat is not None:
                if self.embed_point_feats:
                    sharp_feat = self.embedder(sharp_feat)
                sharp_data = torch.cat([sharp_data, sharp_feat], dim=-1)
            sharp_data = self.input_proj_sharp(sharp_data)

        if self.use_downsample:
            ###### fps
            from torch_cluster import fps

            flattened = pc.view(bs * N, D)  # bs*N, 64

            batch = torch.arange(bs).to(pc.device)
            batch = torch.repeat_interleave(batch, N)  # bs*N

            pos = flattened.to(torch.float16)
            ratio = 1.0 * self.downsmaple_num_latents / N
            idx = fps(pos, batch, ratio=ratio).detach()
            query = data.view(bs * N, -1)[idx].view(bs, -1, data.shape[-1])

            if self.with_sharp_data:
                bs, N, D = sharp_pc.shape
                flattened = sharp_pc.view(bs * N, D)  # bs*N, 64
                pos = flattened.to(torch.float16)
                ratio = 1.0 * self.downsmaple_num_latents / N
                idx = fps(pos, batch, ratio=ratio).detach()
                sharp_query = sharp_data.view(bs * N, -1)[idx].view(
                    bs, -1, sharp_data.shape[-1]
                )
                query = torch.cat([query, sharp_query], dim=1)
        else:
            query = self.query
            query = repeat(query, "m c -> b m c", b=bs)

        latents = self.cross_attn(query, data)
        if self.with_sharp_data:
            latents = latents + self.cross_attn_sharp(query, sharp_data)
        latents = self.self_attn(latents)

        if self.ln_post is not None:
            latents = self.ln_post(latents)

        return latents

    def forward(
        self,
        pc: torch.FloatTensor,
        feats: Optional[torch.FloatTensor] = None,
        sharp_pc: Optional[torch.FloatTensor] = None,
        sharp_feats: Optional[torch.FloatTensor] = None,
    ):
        """

        Args:
            pc (torch.FloatTensor): [B, N, 3]
            feats (torch.FloatTensor or None): [B, N, C]

        Returns:
            dict
        """

        return checkpoint(
            self._forward,
            (pc, feats, sharp_pc, sharp_feats),
            self.parameters(),
            self.use_checkpoint,
        )


class PerceiverCrossAttentionDecoder(ModelMixin, nn.Module):

    def __init__(
        self,
        num_latents: int,
        out_dim: int,
        embedder: FourierEmbedder,
        width: int,
        heads: int,
        init_scale: float = 0.25,
        qkv_bias: bool = True,
        qk_norm: bool = True,
        use_flash: bool = False,
        use_checkpoint: bool = False,
    ):

        super().__init__()

        self.use_checkpoint = use_checkpoint
        self.embedder = embedder

        self.query_proj = nn.Linear(self.embedder.out_dim, width)

        self.cross_attn_decoder = ResidualCrossAttentionBlock(
            n_data=num_latents,
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            use_flash=use_flash,
        )

        self.ln_post = nn.LayerNorm(width)
        self.output_proj = nn.Linear(width, out_dim)

    def _forward(self, queries: torch.FloatTensor, latents: torch.FloatTensor):
        queries = self.query_proj(self.embedder(queries))
        x = self.cross_attn_decoder(queries, latents)
        x = self.ln_post(x)
        x = self.output_proj(x)
        return x

    def forward(self, queries: torch.FloatTensor, latents: torch.FloatTensor):
        return checkpoint(
            self._forward, (queries, latents), self.parameters(), self.use_checkpoint
        )


@step1x3d_geometry.register("michelangelo-autoencoder")
class MichelangeloAutoencoder(BaseModule):
    r"""
    A VAE model for encoding shapes into latents and decoding latent representations into shapes.
    """

    @dataclass
    class Config(BaseModule.Config):
        pretrained_model_name_or_path: str = ""
        subfolder: str = ""
        n_samples: int = 4096
        use_downsample: bool = False
        downsample_ratio: float = 0.0625
        num_latents: int = 256
        point_feats: int = 0
        embed_point_feats: bool = False
        out_dim: int = 1
        embed_dim: int = 64
        embed_type: str = "fourier"
        num_freqs: int = 8
        include_pi: bool = True
        width: int = 768
        heads: int = 12
        num_encoder_layers: int = 8
        num_decoder_layers: int = 16
        init_scale: float = 0.25
        qkv_bias: bool = True
        qk_norm: bool = False
        use_ln_post: bool = False
        use_flash: bool = False
        use_checkpoint: bool = True
        use_multi_reso: Optional[bool] = False
        resolutions: Optional[List[int]] = None
        sampling_prob: Optional[List[float]] = None
        with_sharp_data: Optional[bool] = True
        volume_decoder_type: str = "hierarchical"
        surface_extractor_type: str = "mc"
        z_scale_factor: float = 1.0

    cfg: Config

    def configure(self) -> None:
        super().configure()

        self.embedder = get_embedder(
            embed_type=self.cfg.embed_type,
            num_freqs=self.cfg.num_freqs,
            include_pi=self.cfg.include_pi,
        )

        # encoder
        self.cfg.init_scale = self.cfg.init_scale * math.sqrt(1.0 / self.cfg.width)
        self.encoder = PerceiverCrossAttentionEncoder(
            use_downsample=self.cfg.use_downsample,
            embedder=self.embedder,
            num_latents=self.cfg.num_latents,
            point_feats=self.cfg.point_feats,
            embed_point_feats=self.cfg.embed_point_feats,
            width=self.cfg.width,
            heads=self.cfg.heads,
            layers=self.cfg.num_encoder_layers,
            init_scale=self.cfg.init_scale,
            qkv_bias=self.cfg.qkv_bias,
            qk_norm=self.cfg.qk_norm,
            use_ln_post=self.cfg.use_ln_post,
            use_flash=self.cfg.use_flash,
            use_checkpoint=self.cfg.use_checkpoint,
            use_multi_reso=self.cfg.use_multi_reso,
            resolutions=self.cfg.resolutions,
            sampling_prob=self.cfg.sampling_prob,
            with_sharp_data=self.cfg.with_sharp_data,
        )

        if self.cfg.embed_dim > 0:
            # VAE embed
            self.pre_kl = nn.Linear(self.cfg.width, self.cfg.embed_dim * 2)
            self.post_kl = nn.Linear(self.cfg.embed_dim, self.cfg.width)
            self.latent_shape = (self.cfg.num_latents, self.cfg.embed_dim)
        else:
            self.latent_shape = (self.cfg.num_latents, self.cfg.width)

        self.transformer = Perceiver(
            n_ctx=self.cfg.num_latents,
            width=self.cfg.width,
            layers=self.cfg.num_decoder_layers,
            heads=self.cfg.heads,
            init_scale=self.cfg.init_scale,
            qkv_bias=self.cfg.qkv_bias,
            qk_norm=self.cfg.qk_norm,
            use_flash=self.cfg.use_flash,
            use_checkpoint=self.cfg.use_checkpoint,
        )

        # decoder
        self.decoder = PerceiverCrossAttentionDecoder(
            embedder=self.embedder,
            out_dim=self.cfg.out_dim,
            num_latents=self.cfg.num_latents,
            width=self.cfg.width,
            heads=self.cfg.heads,
            init_scale=self.cfg.init_scale,
            qkv_bias=self.cfg.qkv_bias,
            qk_norm=self.cfg.qk_norm,
            use_flash=self.cfg.use_flash,
            use_checkpoint=self.cfg.use_checkpoint,
        )

        # volume decoder
        if self.cfg.volume_decoder_type == "hierarchical":
            self.volume_decoder = HierarchicalVolumeDecoder()
        else:
            self.volume_decoder = VanillaVolumeDecoder()

        if self.cfg.pretrained_model_name_or_path != "":
            local_model_path = f"{smart_load_model(self.cfg.pretrained_model_name_or_path, self.cfg.subfolder)}/vae/diffusion_pytorch_model.safetensors"
            pretrain_safetensors = load_file(local_model_path)
            print(f"Loading pretrained VAE model from {local_model_path}")

            if "state_dict" in pretrain_safetensors:
                _pretrained_safetensors = {}
                for k, v in pretrain_safetensors["state_dict"].items():
                    if k.startswith("shape_model."):
                        if "proj1" in k:
                            _pretrained_safetensors[
                                k.replace("shape_model.", "").replace(
                                    "proj1", "proj_sharp"
                                )
                            ] = v
                        elif "attn1" in k:
                            _pretrained_safetensors[
                                k.replace("shape_model.", "").replace(
                                    "attn1", "attn_sharp"
                                )
                            ] = v
                        else:
                            _pretrained_safetensors[k.replace("shape_model.", "")] = v

                pretrain_safetensors = _pretrained_safetensors
                self.load_state_dict(pretrain_safetensors, strict=True)
            else:
                _pretrained_safetensors = {}
                for k, v in pretrain_safetensors.items():
                    if k.startswith("shape_model"):
                        final_module = self
                        for key in k.replace("shape_model.", "").split("."):
                            final_module = getattr(final_module, key)
                        data = final_module.data
                        data_zero = torch.zeros_like(data).to(v)

                        if data.shape != v.shape:
                            if data.ndim == 1:
                                data_zero[: v.shape[0]] = v
                            elif data.ndim == 2:
                                data_zero[: v.shape[0], : v.shape[1]] = v
                            v = data_zero

                        _pretrained_safetensors[k.replace("shape_model.", "")] = v
                    else:
                        _pretrained_safetensors[k] = v
                pretrain_safetensors = _pretrained_safetensors
                self.load_state_dict(pretrain_safetensors, strict=True)
                print("Successed load pretrained VAE model")

    def encode(
        self,
        surface: torch.FloatTensor,
        sample_posterior: bool = True,
        sharp_surface: torch.FloatTensor = None,
    ):
        """
        Args:
            surface (torch.FloatTensor): [B, N, 3+C]
            sample_posterior (bool):

        Returns:
            shape_latents (torch.FloatTensor): [B, num_latents, width]
            kl_embed (torch.FloatTensor): [B, num_latents, embed_dim]
            posterior (DiagonalGaussianDistribution or None):
        """
        assert (
            surface.shape[-1] == 3 + self.cfg.point_feats
        ), f"\
            Expected {3 + self.cfg.point_feats} channels, got {surface.shape[-1]}"

        pc, feats = surface[..., :3], surface[..., 3:]  # B, n_samples, 3
        if sharp_surface is not None:
            sharp_pc, sharp_feats = (
                sharp_surface[..., :3],
                sharp_surface[..., 3:],
            )  # B, n_samples, 3
        else:
            sharp_pc, sharp_feats = None, None

        shape_embeds = self.encoder(
            pc, feats, sharp_pc, sharp_feats
        )  # B, num_latents, width
        kl_embed, posterior = self.encode_kl_embed(
            shape_embeds, sample_posterior
        )  # B, num_latents, embed_dim

        kl_embed = kl_embed * self.cfg.z_scale_factor  # encode with scale

        return shape_embeds, kl_embed, posterior

    def decode(self, latents: torch.FloatTensor):
        """
        Args:
            latents (torch.FloatTensor): [B, embed_dim]

        Returns:
            latents (torch.FloatTensor): [B, embed_dim]
        """
        latents = self.post_kl(
            latents / self.cfg.z_scale_factor
        )  # [B, num_latents, embed_dim] -> [B, num_latents, width]

        return self.transformer(latents)

    def query(self, queries: torch.FloatTensor, latents: torch.FloatTensor):
        """
        Args:
            queries (torch.FloatTensor): [B, N, 3]
            latents (torch.FloatTensor): [B, embed_dim]

        Returns:
            features (torch.FloatTensor): [B, N, C], output features
        """

        features = self.decoder(queries, latents)

        return features

    def encode_kl_embed(
        self, latents: torch.FloatTensor, sample_posterior: bool = True
    ):
        posterior = None
        if self.cfg.embed_dim > 0:
            moments = self.pre_kl(latents)
            posterior = DiagonalGaussianDistribution(moments, feat_dim=-1)
            if sample_posterior:
                kl_embed = posterior.sample()
            else:
                kl_embed = posterior.mode()
        else:
            kl_embed = latents
        return kl_embed, posterior

    def forward(
        self,
        surface: torch.FloatTensor,
        sharp_surface: torch.FloatTensor = None,
        rand_points: torch.FloatTensor = None,
        sample_posterior: bool = True,
        **kwargs,
    ):
        shape_latents, kl_embed, posterior = self.encode(
            surface, sample_posterior=sample_posterior, sharp_surface=sharp_surface
        )

        latents = self.decode(kl_embed)  # [B, num_latents, width]

        meshes = self.extract_geometry(latents, **kwargs)

        return shape_latents, latents, posterior, meshes

    def extract_geometry(self, latents: torch.FloatTensor, **kwargs):

        grid_logits_list = []
        for i in range(latents.shape[0]):
            grid_logits = self.volume_decoder(
                latents[i].unsqueeze(0), self.query, **kwargs
            )
            grid_logits_list.append(grid_logits)
        grid_logits = torch.cat(grid_logits_list, dim=0)

        # extract mesh
        surface_extractor_type = (
            kwargs["surface_extractor_type"]
            if "surface_extractor_type" in kwargs.keys()
            and kwargs["surface_extractor_type"] is not None
            else self.cfg.surface_extractor_type
        )

        if surface_extractor_type == "mc":
            surface_extractor = MCSurfaceExtractor()
            meshes = surface_extractor(grid_logits, **kwargs)
        elif surface_extractor_type == "dmc":
            surface_extractor = DMCSurfaceExtractor()
            meshes = surface_extractor(grid_logits, **kwargs)
        else:
            raise NotImplementedError

        return meshes
