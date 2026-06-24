import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int = 8, alpha: float = 1.0, bias: bool = True):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # # 原始线性层（冻结）
        # self.linear = nn.Linear(in_features, out_features, bias=bias)
        # # 冻结原始权重
        # for param in self.linear.parameters():
        #     param.requires_grad = False

        # LoRA 可训练部分
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.reset_parameters()

    def reset_parameters(self):
        # 初始化 A 为高斯，B 为零（标准 LoRA 初始化）
        nn.init.normal_(self.lora_A, std=1 / self.rank)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 原始输出 + LoRA 修正
        return (x @ self.lora_A.T @ self.lora_B.T) * self.scaling