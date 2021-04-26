#!/usr/bin/env python

# Copyright 2021  CASIA  (Authors: Jing Shi)
# Apache 2.0
import argparse
import random
from pathlib import Path

def random_sample(idx,spk_list,audios,num_spk,snr_range=5.0):
    """ Generate one line of the mixtures' list.

    Args:
        idx (int): index in all the mixtures
        spk_list (list): speakers list
        audios (dict): dict with speakers
        num_spk (int): num of speakers in one mixture.
        snr_range (float, optional): [description]. Defaults to 5.0.
    """

    spks = random.sample(spk_list, num_spk)
    line = ""
    snr_pos = round(random.random() * snr_range/2.0, 5) 
    snr_neg = -1 * snr_pos
    for jdx, spk in enumerate(spks):
        line += random.sample(audios[spk],1)[0]
        line += " "
        if jdx == 0:
            line += str(snr_pos)
        elif jdx ==1:
            line += str(snr_neg)
        elif jdx ==2:
            line += str(0)
        line += " "
    return line


def prepare_data(args):
    audiodir = Path(args.vctk_root).expanduser()
    outfile = Path(args.outfile).expanduser().resolve()
    spk_list = [spk_name.name for spk_name in Path(audiodir/"wav48").glob("*")]
    assert len(spk_list)==109, "VCTK should get 109 speakers in total, but got {} now".format(len(spk_list))

    audios = {
        spk: [str(sample.relative_to(audiodir)) for sample in audiodir.rglob(spk + "/*." + args.audio_format)] 
        for spk in spk_list 
    }

    # Shuffle many times to the spk_list
    random.shuffle(spk_list)
    random.shuffle(spk_list)
    random.shuffle(spk_list)
    spk_list_tr = spk_list[:-args.num_spks_test]
    spk_list_tt = spk_list[-args.num_spks_test:]

    with Path(outfile/"spk_list_tr").open("w") as out:
        out.write("\n".join(spk_list_tr))
        out.write("\n")
    with Path(outfile/"spk_list_tt").open("w") as out:
        out.write("\n".join(spk_list_tt))
        out.write("\n")

    for num_spk in args.num_spks:
        for mode in ["tr","cv","tt"]:
            aimfile= "vctk_mix_{}_spk_{}.txt".format(num_spk,mode)
            with Path(outfile/aimfile).open("w") as out:
                for idx_mixture in range(getattr(args,"num_mixtures_{}".format(mode))):
                    # out.write("idx_mixture_{}".format(idx_mixture) + "\n")
                    if mode == "tt": # for open condition
                        aim_list = spk_list_tt
                    else: # for close condition
                        aim_list = spk_list_tr
                    out.write(random_sample(idx_mixture,aim_list,audios,num_spk)+"\n")

    print("Generation of mixture list Finished.")

def get_parser():
    """Argument parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vctk_root", type=str, help="Path to the VCTK root (v0.80)"
    )
    parser.add_argument("--outfile", type=str,default="./")
    parser.add_argument(
        "--num_spks",
        type=int,
        nargs="+",
        required=True,
        help="Number of speakers in one mixture",
    )
    parser.add_argument(
        "--num_spks_test",
        type=int, 
        default=19, 
        help="number of unknwon speakers from total(109)",
    )

    parser.add_argument("--audio-format", type=str, default="wav")
    parser.add_argument("--num_mixtures_tr", type=int, default=20000)
    parser.add_argument("--num_mixtures_cv", type=int, default=5000)
    parser.add_argument("--num_mixtures_tt", type=int, default=3000)
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    prepare_data(args)