import fairseq
import numpy as np
import resampy
import torch

from espnet2.enh.encoder.abs_encoder import AbsEncoder


class Conv16kEncoder(AbsEncoder):
    """Wav2vec2.0 encoder for speech enhancement and separation """

    def __init__(
        self,
        channel: int = 1024,
        kernel_size: int = 640, 
        stride: int = 320,
        pre_rate: int = 16000, 
    ):
        super().__init__()

        self.pre_rate = pre_rate
        self.conv1d = torch.nn.Conv1d(
            1, channel, kernel_size=kernel_size, stride=stride, bias=False
        )
        self._output_dim = channel

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, input: torch.Tensor, ilens: torch.Tensor):
        """Forward.

        Args:
            input (torch.Tensor): mixed speech [Batch, sample]
            ilens (torch.Tensor): input lengths [Batch]
        Returns:
            feature (torch.Tensor): mixed feature after encoder [Batch, flens, channel]
        """
        assert input.dim() == 2, "Currently only support single channle input"

        # input = torch.unsqueeze(input, 1)

        # [Batch, sample] --> [Batch, flens, channel]
        # print('before resample:', input.shape)
        if self.pre_rate != 8000:
            wav_new = resampy.resample(
                input.data.cpu().numpy().astype(np.float64).T, 8000, self.pre_rate, axis=0
            )
            wav = torch.tensor(wav_new).to(input.device).float().transpose(0, 1)
        feature = self.conv1d(wav.unsqueeze(1)).transpose(1,2)

        # feature = torch.nn.functional.relu(feature)

        flens = feature.shape[1]

        return feature, flens
