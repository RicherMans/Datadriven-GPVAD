import ignite.metrics as metrics
import torch
import torch.nn as nn


class FrameBCELoss(nn.Module):
    """docstring for BCELoss"""
    def __init__(self):
        super().__init__()

    def forward(self, clip_prob, frame_prob, tar_time, tar_clip, length):
        batchsize, timesteps, ndim = tar_time.shape
        idxs = torch.arange(timesteps, device='cpu').repeat(batchsize).view(
            batchsize, timesteps)
        mask = (idxs < length.view(-1, 1)).to(frame_prob.device)
        masked_bce = nn.functional.binary_cross_entropy(
            input=frame_prob, target=tar_time,
            reduction='none') * mask.unsqueeze(-1)
        return masked_bce.sum() / mask.sum()


class ClipFrameBCELoss(nn.Module):
    """docstring for BCELoss"""
    def __init__(self):
        super().__init__()
        self.frameloss = FrameBCELoss()
        self.cliploss = nn.BCELoss()

    def forward(self, clip_prob, frame_prob, tar_time, tar_clip, length):
        return self.frameloss(
            clip_prob, frame_prob, tar_time, tar_clip, length) + self.cliploss(
                clip_prob, tar_clip)


class BCELossWithLabelSmoothing(nn.Module):
    """docstring for BCELoss"""
    def __init__(self, label_smoothing=0.1):
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(self, clip_prob, frame_prob, tar):
        n_classes = clip_prob.shape[-1]
        with torch.no_grad():
            tar = tar * (1 - self.label_smoothing) + (
                1 - tar) * self.label_smoothing / (n_classes - 1)
        return nn.functional.binary_cross_entropy(clip_prob, tar)


# Reimplement Loss, because ignite loss only takes 2 args, not 3 and nees to parse kwargs around ... just *output does the trick
class Loss(metrics.Loss):
    def __init__(self,
                 loss_fn,
                 output_transform=lambda x: x,
                 batch_size=lambda x: len(x),
                 device=None):
        super(Loss, self).__init__(loss_fn=loss_fn,
                                   output_transform=output_transform,
                                   batch_size=batch_size)

    def update(self, output):
        average_loss = self._loss_fn(*output)

        if len(average_loss.shape) != 0:
            raise ValueError('loss_fn did not return the average loss.')

        N = self._batch_size(output[0])
        self._sum += average_loss.item() * N
        self._num_examples += N


if __name__ == "__main__":
    batch, time, dim = 4, 500, 10
    frame = torch.sigmoid(torch.randn(batch, time, dim))
    clip = torch.sigmoid(torch.randn(batch, dim))
    tar = torch.empty(batch, dim).random_(2)
