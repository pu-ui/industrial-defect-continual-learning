from __future__ import annotations

import copy
from typing import Iterable

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50


class CoordinateAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 32):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.reduce = nn.Conv2d(channels, hidden, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(hidden)
        self.activation = nn.Hardswish()
        self.height_gate = nn.Conv2d(hidden, channels, kernel_size=1)
        self.width_gate = nn.Conv2d(hidden, channels, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        height, width = inputs.shape[-2:]
        height_context = inputs.mean(dim=3, keepdim=True)
        width_context = inputs.mean(dim=2, keepdim=True).transpose(2, 3)
        context = torch.cat([height_context, width_context], dim=2)
        context = self.activation(self.bn(self.reduce(context)))
        height_context, width_context = torch.split(context, [height, width], dim=2)
        width_context = width_context.transpose(2, 3)
        height_weight = self.height_gate(height_context).sigmoid()
        width_weight = self.width_gate(width_context).sigmoid()
        return inputs * height_weight * width_weight


class CoordinateAttentionBottleneck(nn.Module):
    """Apply coordinate attention to the residual branch before skip addition."""

    def __init__(self, block: nn.Module, output_channels: int):
        super().__init__()
        self.block = block
        self.attention = CoordinateAttention(output_channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        identity = inputs
        outputs = self.block.conv1(inputs)
        outputs = self.block.bn1(outputs)
        outputs = self.block.relu(outputs)
        outputs = self.block.conv2(outputs)
        outputs = self.block.bn2(outputs)
        outputs = self.block.relu(outputs)
        outputs = self.block.conv3(outputs)
        outputs = self.block.bn3(outputs)
        outputs = self.attention(outputs)
        if self.block.downsample is not None:
            identity = self.block.downsample(inputs)
        outputs += identity
        outputs = self.block.relu(outputs)
        return outputs


class CoordinateAttentionStage(nn.Sequential):
    def __init__(self, stage: nn.Sequential, output_channels: int):
        super().__init__(
            *(
                CoordinateAttentionBottleneck(block, output_channels)
                for block in stage.children()
            )
        )


class CoordinateAttentionResNet50(nn.Module):
    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        use_coordinate_attention: bool = True,
    ):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = resnet50(weights=weights)
        self.use_coordinate_attention = use_coordinate_attention
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        if use_coordinate_attention:
            self.layer2 = CoordinateAttentionStage(backbone.layer2, 512)
            self.layer3 = CoordinateAttentionStage(backbone.layer3, 1024)
            self.layer4 = CoordinateAttentionStage(backbone.layer4, 2048)
        else:
            self.layer2 = backbone.layer2
            self.layer3 = backbone.layer3
            self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool
        self.fc = nn.Linear(backbone.fc.in_features, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.maxpool(self.relu(self.bn1(self.conv1(inputs))))
        outputs = self.layer1(outputs)
        outputs = self.layer2(outputs)
        outputs = self.layer3(outputs)
        outputs = self.layer4(outputs)
        outputs = self.avgpool(outputs)
        return self.fc(torch.flatten(outputs, 1))

    def freeze_for_continual_learning(self) -> None:
        """PECL policy: train CA modules, BN affine terms, and classifier only."""
        for parameter in self.parameters():
            parameter.requires_grad = False
        for module in self.modules():
            if isinstance(module, CoordinateAttention):
                for parameter in module.parameters():
                    parameter.requires_grad = True
            elif isinstance(module, nn.BatchNorm2d):
                if module.weight is not None:
                    module.weight.requires_grad = True
                if module.bias is not None:
                    module.bias.requires_grad = True
        for parameter in self.fc.parameters():
            parameter.requires_grad = True

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        return (parameter for parameter in self.parameters() if parameter.requires_grad)

    def frozen_copy(self) -> "CoordinateAttentionResNet50":
        old_model = copy.deepcopy(self).eval()
        for parameter in old_model.parameters():
            parameter.requires_grad = False
        return old_model


def parameter_counts(model: nn.Module) -> dict[str, int]:
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }
