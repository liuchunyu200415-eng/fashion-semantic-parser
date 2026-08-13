"""Query-conditioned multiscale decoder for frozen DINOv2 patch features."""

from typing import Any


def build_multiscale_patch_decoder(
    feature_dimension: int,
    settings: Any,
) -> Any:
    """Build a lightweight query-conditioned spatial patch decoder.

    Args:
        feature_dimension: Positive shared DINOv2/text feature dimension.
        settings: Dense settings with hidden, branch, dilation, and dropout values.

    Returns:
        A PyTorch module mapping fused patch/query grids to one logit grid.

    Raises:
        ValueError: If feature dimension or configured decoder geometry is invalid.
        RuntimeError: If PyTorch is unavailable.
    """
    if feature_dimension < 1:
        raise ValueError("Decoder feature dimension must be positive.")
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError("PyTorch is required for the patch decoder.") from error
    hidden = int(settings.decoder_hidden_dimension)
    branch = int(settings.decoder_branch_dimension)
    dilations = tuple(int(value) for value in settings.decoder_dilations)
    if hidden % 8 or branch % 8 or not dilations:
        raise ValueError("Patch decoder dimensions or dilations are invalid.")
    input_channels = feature_dimension * 3 + 3

    class _MultiscalePatchDecoder(torch.nn.Module):
        """Fuse visual, language, interaction, coordinate, and context signals."""

        def __init__(self) -> None:
            super().__init__()
            self.stem = torch.nn.Sequential(
                torch.nn.Conv2d(input_channels, hidden, kernel_size=1),
                torch.nn.GroupNorm(8, hidden),
                torch.nn.GELU(),
            )
            self.branches = torch.nn.ModuleList(
                [
                    torch.nn.Sequential(
                        torch.nn.Conv2d(
                            hidden,
                            branch,
                            kernel_size=3,
                            padding=dilation,
                            dilation=dilation,
                        ),
                        torch.nn.GroupNorm(8, branch),
                        torch.nn.GELU(),
                    )
                    for dilation in dilations
                ]
            )
            self.output = torch.nn.Sequential(
                torch.nn.Conv2d(branch * len(dilations), branch, kernel_size=1),
                torch.nn.GELU(),
                torch.nn.Dropout2d(float(settings.decoder_dropout)),
                torch.nn.Conv2d(branch, 1, kernel_size=1),
            )

        def forward(self, fused_features: Any) -> Any:
            """Decode one fused ``BxCxHxW`` feature grid."""
            hidden_features = self.stem(fused_features)
            contexts = [branch_model(hidden_features) for branch_model in self.branches]
            return self.output(torch.cat(contexts, dim=1)).squeeze(1)

    return _MultiscalePatchDecoder()


def multiscale_patch_decoder_logits(
    decoder: Any,
    patch_features: Any,
    projected_text: Any,
    calibration: tuple[Any, Any, float],
) -> Any:
    """Return spatially decoded query-conditioned patch logits.

    Args:
        decoder: Trainable multiscale decoder module.
        patch_features: Tensor shaped ``BxPxD``.
        projected_text: Tensor shaped ``BxD``.
        calibration: ``(log_scale, logit_bias, maximum_scale)`` values.

    Returns:
        Calibrated logits shaped ``BxP``.

    Raises:
        ValueError: If batch, feature, patch, or grid geometry is inconsistent.
        RuntimeError: If PyTorch is unavailable.
    """
    patch_count = int(patch_features.shape[1]) if patch_features.ndim == 3 else 0
    grid_size = round(patch_count**0.5)
    if (
        patch_features.ndim != 3
        or projected_text.ndim != 2
        or patch_features.shape[0] != projected_text.shape[0]
        or patch_features.shape[2] != projected_text.shape[1]
        or grid_size < 1
        or patch_count != grid_size * grid_size
    ):
        raise ValueError("Multiscale patch and text feature geometry is invalid.")
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError("PyTorch is required for patch decoder logits.") from error
    patches = torch.nn.functional.normalize(patch_features.float(), dim=2)
    queries = torch.nn.functional.normalize(projected_text.float(), dim=1)
    batch_size, _, feature_dimension = patches.shape
    patch_grid = patches.transpose(1, 2).reshape(
        batch_size,
        feature_dimension,
        grid_size,
        grid_size,
    )
    query_grid = queries.reshape(batch_size, feature_dimension, 1, 1).expand(
        -1,
        -1,
        grid_size,
        grid_size,
    )
    interaction = patch_grid * query_grid
    cosine = interaction.sum(dim=1, keepdim=True)
    coordinates = _coordinate_grid(
        batch_size,
        grid_size,
        device=patches.device,
        dtype=patches.dtype,
    )
    fused = torch.cat(
        [patch_grid, query_grid, interaction, cosine, coordinates],
        dim=1,
    )
    raw_logits = decoder(fused).reshape(batch_size, -1)
    log_scale, logit_bias, max_logit_scale = calibration
    scale = torch.clamp(log_scale.exp(), max=max_logit_scale)
    return scale * raw_logits + logit_bias


def _coordinate_grid(
    batch_size: int,
    grid_size: int,
    *,
    device: Any,
    dtype: Any,
) -> Any:
    """Return normalized image-frame ``x/y`` coordinates for each batch row."""
    import torch  # type: ignore[import-not-found]

    values = torch.linspace(-1.0, 1.0, grid_size, device=device, dtype=dtype)
    y_grid, x_grid = torch.meshgrid(values, values, indexing="ij")
    coordinates = torch.stack([x_grid, y_grid], dim=0).unsqueeze(0)
    return coordinates.expand(batch_size, -1, -1, -1)
