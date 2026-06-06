"""PyTorch policy network for move prediction (behavioral cloning).

Input : (batch, 12, 8, 8) board tensors (side-to-move perspective).
Output: (batch, 4096) logits over (from, to) square pairs.

A small residual CNN -- big enough to learn opening/middlegame habits, small enough
to train on CPU in a reasonable time.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding import BOARD_PLANES, NUM_MOVES


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class PolicyNet(nn.Module):
    def __init__(self, channels=128, num_blocks=4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(BOARD_PLANES, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_blocks)]
        )
        # Policy head: 1x1 conv down to 32 planes, then a linear map to 4096 logits.
        self.policy_conv = nn.Sequential(
            nn.Conv2d(channels, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.policy_fc = nn.Linear(32 * 8 * 8, NUM_MOVES)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.policy_conv(x)
        x = x.flatten(1)
        return self.policy_fc(x)
