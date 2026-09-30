"""
Backward-compatibility facade for diagnostic model CNN architectures.
Re-exports from anemia_ai.models.cnn_factory.
"""

from anemia_ai.models.cnn_factory import create_cnn_backbone


def get_model(architecture: str = "efficientnet_b0", pretrained: bool = True):
    return create_cnn_backbone(architecture=architecture, pretrained=pretrained)


__all__ = ["get_model", "create_cnn_backbone"]
