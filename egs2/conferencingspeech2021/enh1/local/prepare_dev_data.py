#!/usr/bin/env python

# Copyright 2021  Shanghai Jiao Tong University (Authors: Wangyou Zhang)
# Apache 2.0
import argparse
from pathlib import Path

from espnet2.fileio.datadir_writer import DatadirWriter


def prepare_data(args):
    config_file = Path(args.config_file).expanduser().resolve()
    audiodirs = [Path(audiodir).expanduser().resolve() for audiodir in args.audiodirs]
    audios = {
        path.stem: str(path)
        for audiodir in audiodirs
        for path in audiodir.rglob("*.wav")
    }
    with DatadirWriter(args.outdir) as writer, config_file.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # /path/SSB18100388.wav -2 /path/noise-free-sound-0328.wav /path/circle/3.43_5.92_3.00_1.75_2.50_184.5997_262.1617_0.6728.wav 19.249479746268474 0.44975649854951305
            path_clean, start_time, path_noise, path_rir, snr, scale = line.split()
            uttid = "#".join(
                [
                    Path(path_clean).stem,
                    Path(path_noise).stem,
                    Path(path_rir).stem,
                    start_time,
                    snr,
                    scale,
                ]
            )
            writer["wav.scp"][uttid] = audios[uttid]
            writer["spk1.scp"][uttid] = path_clean
            if "librispeech" in path_clean:
                spkid = "-".join(path_clean.split("/")[-3:-1])
            else:
                spkid = path_clean.split("/")[-2]
            writer["utt2spk"][uttid] = spkid
            writer["noise1.scp"][uttid] = path_noise


def get_parser():
    """Argument parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_file", type=str, help="Path to the list of audio files for training"
    )
    parser.add_argument(
        "--audiodirs",
        type=str,
        nargs="+",
        required=True,
        help="Paths to the directories containing simulated audio files",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="Paths to the directory for storing *.scp, utt2spk, spk2utt",
    )
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    prepare_data(args)
