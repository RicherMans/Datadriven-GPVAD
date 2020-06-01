import torch
import logging
import torch.nn as nn
import numpy as np


class RandomPad(nn.Module):
    """docstring for RandomPad"""
    def __init__(self, value=0., padding=0):
        super().__init__()
        self.value = value
        self.padding = padding

    def forward(self, x):
        if self.training and self.padding > 0:
            left_right = torch.empty(2).random_(self.padding).int().numpy()
            topad = (0, 0, *left_right)
            x = nn.functional.pad(x, topad, value=self.value)
        return x


class Roll(nn.Module):
    """docstring for Roll"""
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        if self.training:
            shift = torch.empty(1).normal_(self.mean, self.std).int().item()
            x = torch.roll(x, shift, dims=0)
        return x


class RandomCrop(nn.Module):
    """docstring for RandomPad"""
    def __init__(self, size: int = 100):
        super().__init__()
        self.size = int(size)

    def forward(self, x):
        if self.training:
            time, freq = x.shape
            if time < self.size:
                return x
            hi = time - self.size
            start_ind = torch.empty(1, dtype=torch.long).random_(0, hi).item()
            x = x[start_ind:start_ind + self.size, :]
        return x


class TimeMask(nn.Module):
    def __init__(self, n=1, p=50):
        super().__init__()
        self.p = p
        self.n = 1

    def forward(self, x):
        time, freq = x.shape
        if self.training:
            for i in range(self.n):
                t = torch.empty(1, dtype=int).random_(self.p).item()
                to_sample = max(time - t, 1)
                t0 = torch.empty(1, dtype=int).random_(to_sample).item()
                x[t0:t0 + t, :] = 0
        return x


class FreqMask(nn.Module):
    def __init__(self, n=1, p=12):
        super().__init__()
        self.p = p
        self.n = 1

    def forward(self, x):
        time, freq = x.shape
        if self.training:
            for i in range(self.n):
                f = torch.empty(1, dtype=int).random_(self.p).item()
                f0 = torch.empty(1, dtype=int).random_(freq - f).item()
                x[:, f0:f0 + f] = 0.
        return x


class GaussianNoise(nn.Module):
    """docstring for Gaussian"""
    def __init__(self, snr=30, mean=0):
        super().__init__()
        self._mean = mean
        self._snr = snr

    def forward(self, x):
        if self.training:
            E_x = (x**2).sum()/x.shape[0]
            noise = torch.empty_like(x).normal_(self._mean, std=1)
            E_noise = (noise**2).sum()/noise.shape[0]
            alpha = np.sqrt(E_x / (E_noise * pow(10, self._snr / 10)))
            x = x + alpha * noise
        return x


class Shift(nn.Module):
    """
    Randomly shift audio in time by up to `shift` samples.
    """
    def __init__(self, shift=4000):
        super().__init__()
        self.shift = shift

    def forward(self, wav):
        time, channels = wav.size()
        length = time - self.shift
        if self.shift > 0:
            if not self.training:
                wav = wav[..., :length]
            else:
                offset = torch.randint(self.shift, [channels, 1],
                                       device=wav.device)
                indexes = torch.arange(length, device=wav.device)
                offset = indexes + offset
                wav = wav.gather(0, offset.transpose(0, 1))
        return wav


class FlipSign(nn.Module):
    """
    Random sign flip.
    """
    def forward(self, wav):
        time, channels = wav.size()
        if self.training:
            signs = torch.randint(2, (1, channels),
                                  device=wav.device,
                                  dtype=torch.float32)
            wav = wav * (2 * signs - 1)
        return wav


if __name__ == "__main__":
    x = torch.randn(1, 10)
    y = GaussianNoise(10)(x)
    print(x)
    print(y)
