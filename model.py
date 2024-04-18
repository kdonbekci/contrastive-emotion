from torch import nn
import torch
from transformers import WhisperModel, WhisperConfig


class Classifier(nn.Module):
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        max_duration: float,
        projection_dim: int,
        num_classes: int,
        freeze_conv_pos: bool = True,
    ):
        super().__init__()
        self.backbone = WhisperBackbone(
            pretrained_model_name_or_path,
            max_duration,
            pooling="none",
            freeze_conv_pos=freeze_conv_pos,
        )

        self.projection = nn.Linear(
            in_features=self.backbone.encoder.config.d_model,
            out_features=projection_dim,
        )
        self.classification_head = nn.Linear(
            in_features=projection_dim, out_features=num_classes
        )

    def forward(
        self, input_values: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        out = self.backbone(input_values, attention_mask=attention_mask)
        out = self.projection(out)
        output_attention_mask = attention_mask[
            :, ::2, None
        ]  # second conv has stride of 2, so drop half

        # pool over time, weighted by attention mask
        out = out * output_attention_mask
        out = out.sum(dim=1, keepdim=True) / output_attention_mask.sum(
            dim=1, keepdim=True
        )
        out = out.squeeze(1)
        # output classification logits
        out = self.classification_head(out)
        return out

    def freeze_backbone(self, include_conv_pos=False):
        self.backbone.freeze(include_conv_pos=include_conv_pos)

    def unfreeze_backbone(self):
        self.backbone.unfreeze()

    @property
    def device(self):
        return self.backbone.encoder.device


class WhisperBackbone(nn.Module):
    def __init__(
        self,
        pretrained_model_name_or_path,
        max_duration,
        pooling="mean",
        freeze_conv_pos=True,
    ):
        super().__init__()
        assert pooling in ["mean", "none"]
        self.pooling = pooling
        model = WhisperModel.from_pretrained(pretrained_model_name_or_path)
        state_dict = model.state_dict()
        offset = 3000 * (max_duration / 30 / 2)
        assert offset % 1 == 0, f"Chosen {max_duration} yields incomplete frame"
        offset = int(offset)
        state_dict["encoder.embed_positions.weight"] = state_dict[
            "encoder.embed_positions.weight"
        ][:offset, :]
        config = WhisperConfig.from_pretrained(
            pretrained_model_name_or_path, max_source_positions=offset
        )
        model = WhisperModel(config)
        model.load_state_dict(state_dict)
        self.encoder = model.get_encoder()
        if freeze_conv_pos:
            self.freeze(only_conv_pos=True)

    def unfreeze(self, include_conv_pos=False):
        for name, param in self.encoder.named_parameters():
            if not include_conv_pos:
                if name.startswith("embed_positions") or name.startswith("conv"):
                    continue
            param.requires_grad = True

    def freeze(self, only_conv_pos: bool = False):
        for name, param in self.encoder.named_parameters():
            if only_conv_pos:
                if not name.startswith("embed_positions") and not name.startswith(
                    "conv"
                ):
                    continue
            param.requires_grad = False

    def forward(
        self, input_values: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        out = self.encoder(
            input_values, attention_mask=attention_mask
        ).last_hidden_state
        if self.pooling == "mean":
            output_attention_mask = attention_mask[
                :, ::2, None
            ]  # second conv has stride of 2, so drop half
            out = out * output_attention_mask
            out = out.sum(dim=1, keepdim=True) / output_attention_mask.sum(
                dim=1, keepdim=True
            )
            out = out.squeeze()
        return out

    @property
    def device(self):
        return self.encoder.device
