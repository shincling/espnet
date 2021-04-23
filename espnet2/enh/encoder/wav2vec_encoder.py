import fairseq
import numpy as np
import resampy
import torch

from espnet2.enh.encoder.abs_encoder import AbsEncoder


class Wav2vecEncoder(AbsEncoder):
    """Wav2vec2.0 encoder for speech enhancement and separation """

    def __init__(
        self,
        checkpoint: str,
        channel: int = 1024,
        kernel_size: int = None,
        stride: int = None,
        pre_rate: int = 8000, 
        fusing: bool = True,
    ):
        super().__init__()

        checkpoint = torch.load(checkpoint) 
        wav2vec_encoder = fairseq.models.wav2vec.Wav2Vec2Model.build_model(checkpoint['cfg']['model'])
        wav2vec_encoder.load_state_dict(checkpoint['model'])
        self.pre_rate = pre_rate

        self.wav2vec_conv1d = wav2vec_encoder
        self.conv1d = torch.nn.Conv1d(
            1, channel, kernel_size=kernel_size, stride=stride, bias=False
        )
        wav2vec_params = list(self.wav2vec_conv1d.parameters())
        for p in wav2vec_params:
            p.requires_grad = False

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
        device="cuda"
        assert input.dim() == 2, "Currently only support single channle input"
        self.conv1d.to(device)

        with torch.no_grad():
        # input = torch.unsqueeze(input, 1)
        # [Batch, sample] --> [Batch, flens, channel]
        # print('before resample:', input.shape)
            if self.pre_rate != 8000:
                wav_new = resampy.resample(
                    input.data.cpu().numpy().astype(np.float64).T, 8000, self.pre_rate, axis=0
                )
                wav = torch.tensor(wav_new).to(device).float().transpose(0, 1)
                feature_wav2vec = self.wav2vec_conv1d(wav, features_only=True, mask=False)['x']
                # print('after resample:', wav.shape, feature.shape)
            else:
                feature_wav2vec = self.wav2vec_conv1d(input, features_only=True, mask=False)['x']
            feature = self.conv1d(wav.unsqueeze(1)).transpose(1,2)

            assert feature.shape == feature_wav2vec.shape, (input.shape, feature.shape, feature_wav2vec.shape)
            feature = (feature + feature_wav2vec)/2.0

        # feature = torch.nn.functional.relu(feature)

        flens = feature.shape[1]

        return feature, flens
