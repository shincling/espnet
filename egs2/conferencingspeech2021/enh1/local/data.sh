#!/bin/bash

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

help_message=$(cat << EOF
Usage: $0 [--stage <stage>] [--stop_stage <stop_stage>] --official-data-dir <official_data_dir>

  required argument:
    --official-data-dir: path to the directory of offical data for ConferencingSpeech2021 with the following structure:

        <official_data_dir>
         |-- Development_test_set/
         |   |-- playback+noise/
         |   |-- readme.txt
         |   |-- realrecording_cut/
         |   |-- semireal+noise/
         |   \-- simu_single_MA/
         |
         |-- Training_set/
         |   |-- circle_rir/
         |   |-- linear_rir/
         |   |-- non_uniform_linear_rir/
         |   |-- readme.txt
         |   |-- selected_lists/
         |   \-- train_record_noise/
         |
         \-- config_files_simulation_train/
             |-- train_simu_circle.config
             |-- train_simu_linear.config
             \-- train_simu_non_uniform.config

  optional argument:
    [--stage]: 1 (default) or 3
    [--stop_stage]: 1 or 3 (default)
EOF
)


stage=1
stop_stage=3
official_data_dir=

log "$0 $*"
. utils/parse_options.sh


. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

if [ $# -gt 0 ]; then
    log "${help_message}"
    exit 2
fi

if [ ! -e "${official_data_dir}" ]; then
    log "${help_message}"
    log "No such directory for --official-data-dir: '${official_data_dir}'"
    exit 1
fi

if [ ! -e "${AISHELL}" ]; then
    log "Fill the value of 'AISHELL' in db.sh"
    log "(available at http://openslr.org/33/)"
    exit 1
fi

if [ ! -e "${AISHELL3}" ]; then
    log "Fill the value of 'AISHELL3' in db.sh"
    log "(available at http://openslr.org/93/)"
    exit 1
fi

if [ ! -e "${LIBRISPEECH}" ]; then
    log "Fill the value of 'LIBRISPEECH' in db.sh"
    log "(available at http://openslr.org/12/)"
    exit 1
elif [ ! -e "${LIBRISPEECH}/train-clean-360" ]; then
    log "Please ensure '${LIBRISPEECH}/train-clean-360' exists"
    exit 1
fi

if [ ! -e "${VCTK}" ]; then
    log "Fill the value of 'VCTK' in db.sh"
    log "(Version 0.80, available at https://datashare.ed.ac.uk/handle/10283/2651)"
    exit 1
fi

if [ ! -e "${MUSAN}" ]; then
    log "Fill the value of 'MUSAN' in db.sh"
    log "(available at http://openslr.org/17/)"
    exit 1
fi

if [ ! -e "${AUDIOSET}" ]; then
    log "Fill the value of 'AUDIOSET' in db.sh"
    log "(available at https://github.com/marc-moreaux/audioset_raw)"
    exit 1
fi


odir="${PWD}/local"
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Prepare Training and Dev Data for Simulation"

    if [ ! -d "${odir}/ConferencingSpeech2021" ]; then
        git clone https://github.com/ConferencingSpeech/ConferencingSpeech2021.git "${odir}/ConferencingSpeech2021"
    fi
    (
        cd "${odir}/ConferencingSpeech2021"
        # This patch is for simulation/mix_wav.py at commit 49d3b2fc47
        git checkout 49d3b2fc47
        git apply "${odir}/fix_simulation_script.patch"
        python -m pip install -r requirements.txt
    )

    rir_dir="${official_data_dir}/Training_set"
    # if [ "${rir_dir,,}" = "none" ] || [ ! -d "${rir_dir}" ]; then
    #     # Simulate RIRs if not provided
    #     if python -c 'import sys; assert(float(".".join(map(str, sys.version_info[:2]))) < 3.6)' 2>/dev/null; then
    #         log "Python 3.6+ is required for simulation."
    #         exit 2
    #     fi
    #     # This takes ~3.3 hours with Intel(R) Xeon(R) CPU E5-2670 v2 @ 2.50GHz
    #     # ~105 RIRs will be generated in ${odir}/ConferencingSpeech2021/simulation/tmp*.wav
    #     log "Start RIR simulation..."
    #     (
    #         export LD_LIBRARY_PATH="${odir}/ConferencingSpeech2021/simulation":$LD_LIBRARY_PATH
    #         cd "${odir}/ConferencingSpeech2021/simulation"
    #         python ./challenge_rirgenerator.py
    #     )
    #     rir_dir="${odir}/ConferencingSpeech2021/simulation"
    # fi

    # make symbolic links for each corpus to match the data preparation script
    corpora_dir="${odir}/ConferencingSpeech2021/corpora"
    mkdir -p "${corpora_dir}"
    ln -s "${AISHELL}" "${corpora_dir}/aishell_1"
    ln -s "${AISHELL3}" "${corpora_dir}/aishell_3"
    ln -s "${VCTK}" "${corpora_dir}/vctk"
    ln -s "${LIBRISPEECH}/train-clean-360" "${corpora_dir}/librispeech_360"
    ln -s "${MUSAN}" "${corpora_dir}/musan"
    ln -s "${AUDIOSET}" "${corpora_dir}/audioset"
    ln -s "${rir_dir}/linear_rir" "${corpora_dir}/linear"
    ln -s "${rir_dir}/circle_rir" "${corpora_dir}/circle"
    ln -s "${rir_dir}/non_uniform_linear_rir" "${corpora_dir}/non_uniform"

    sed -i -e "s#aishell_1='.*'#aishell_1='${corpora_dir}/aishell_1'#g" \
        -e "s#aishell_3='.*'#aishell_3='${corpora_dir}/aishell_3'#g" \
        -e "s#vctk='.*'#vctk='${corpora_dir}/vctk'#g" \
        -e "s#librispeech='.*'#librispeech='${corpora_dir}/librispeech_360'#g" \
        -e "s#musan='.*'#musan='${corpora_dir}/musan'#g" \
        -e "s#audioset='.*'#audioset='${corpora_dir}/audioset'#g" \
        -e "s#linear='.*'#linear='${corpora_dir}/linear'#g" \
        -e "s#circle='.*'#circle='${corpora_dir}/circle'#g" \
        -e "s#non_uniform='.*'#non_uniform='${corpora_dir}/non_uniform'#g" \
        -e "s#find \${name_path} #find \${name_path}/ #g" \
        "${odir}/ConferencingSpeech2021/simulation/prepare.sh"

    # This script will generate ${odir}/ConferencingSpeech2021/simulation/data/{train,dev}_*.config
    (
        cd "${odir}/ConferencingSpeech2021/simulation"
        # NOTE (wangyou): 1000+ samples in ConferencingSpeech2021/selected_list/train/audioset.name
        # might be unavailable from YouTube due to violation of policies, copyright, and other causes.
        # In this case, you may want to remove them from the list.
        bash ./prepare.sh
    )

    # Prepare the simulated RIR lists for training and development, in case that
    # ${odir}/ConferencingSpeech2021/simulation/prepare.sh fails to finish lines 48-55.
    for name in linear circle non_uniform; do
        for mode in train dev; do
            python local/prepare_data_list.py \
                --outfile "${odir}/ConferencingSpeech2021/simulation/data/${mode}_${name}_rir.lst" \
                --audiodirs "${corpora_dir}/${name}" \
                --audio-format "wav" \
                "${odir}/ConferencingSpeech2021/selected_lists/${mode}/${name}.name"
        done
    done

    # Fill ${odir}/ConferencingSpeech2021/simulation/data/dev_*.config with real paths
    simu_data_path="${odir}/ConferencingSpeech2021/simulation/data"
    for name in linear circle non_uniform; do
        python local/prepare_simu_config.py \
            "${simu_data_path}/dev_${name}_simu_mix.config" \
            --clean_list "${simu_data_path}/dev_clean.lst" \
            --noise_list "${simu_data_path}/dev_noise.lst" \
            --rir_list "${simu_data_path}/dev_${name}_rir.lst" \
            --outfile "${simu_data_path}/dev_${name}_simu_mix.config"
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Data Simulation"

    # Expected data to be generated:
    # ${odir}/ConferencingSpeech2021/simulation/data/wav/dev/
    #  |-- simu_circle/
    #  |   |-- dev_circle_simu_mix.config
    #  |   |-- mix/*.wav             (1588 samples * 8 ch * 6 sec)
    #  |   |-- noreverb_ref/*.wav    (1588 samples * 8 ch * 6 sec)
    #  |   \-- reverb_ref/*.wav      (1588 samples * 8 ch * 6 sec)
    #  |-- simu_linear/
    #  |   |-- dev_linear_simu_mix.config
    #  |   |-- mix/*.wav             (1588 samples * 8 ch * 6 sec)
    #  |   |-- noreverb_ref/*.wav    (1588 samples * 8 ch * 6 sec)
    #  |   \-- reverb_ref/*.wav      (1588 samples * 8 ch * 6 sec)
    #  \-- simu_non_uniform/
    #      |-- dev_non_uniform_simu_mix.config
    #      |-- mix/*.wav             (1588 samples * 8 ch * 6 sec)
    #      |-- noreverb_ref/*.wav    (1588 samples * 8 ch * 6 sec)
    #      \-- reverb_ref/*.wav      (1588 samples * 8 ch * 6 sec)
    (
        cd "${odir}/ConferencingSpeech2021/simulation"
        for name in linear circle non_uniform; do
            log "Simulating with dev_${name}_simu_mix.config"
            python mix_wav.py \
                --mix_config_path data/dev_${name}_simu_mix.config \
                --save_dir data/wavs/dev/simu_${name}/ \
                --chunk_len 6 \
                --generate_config False
        done
    )
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "stage 3: Prepare data directory"

    tmpdir=$(mktemp -d /tmp/conferencingspeech.XXXX)
    ##############################################
    # Training data will be generated on the fly #
    ##############################################
    mkdir -p data/train
    simu_data_path="${odir}/ConferencingSpeech2021/simulation/data"

    # Prepare wav.scp and spk1.scp
    sed -e 's/\.\(wav\|flac\)//' "${simu_data_path}/train_clean.lst" | \
        awk -F '/' '{print $NF}' > "${tmpdir}/utt_clean.list"
    paste -d' ' "${tmpdir}/utt_clean.list" "${simu_data_path}/train_clean.lst" | sort -u > data/train/wav.scp
    cp data/train/wav.scp data/train/spk1.scp

    # Prepare utt2spk for data from aishell_1, aishell_3, librispeech_360, and vctk
    # path -> spkid (aishell_1): .../S0724/BAC009S0724W0121.wav -> S0724
    # path -> spkid (aishell_3): .../SSB0261/SSB02610250.wav -> SSB0261
    # path -> spkid (librispeech_360): .../7932/93470/7932-93470-0006.flac -> 7932-93470
    # path -> spkid (vctk): .../p278/p278_202.wav -> p278
    sed -e 's/\.\(wav\|flac\)//' "${simu_data_path}/train_clean.lst" | \
        awk 'BEGIN{ FS="/" } {
            if(match($0, "librispeech_360")) {i=NF-2; j=NF-1; printf("%s %s-%s\n",$NF,$i,$j)}
            else {i=NF-1; printf("%s %s\n",$NF,$i)}
        }' | sort -u > data/train/utt2spk
    utils/utt2spk_to_spk2utt.pl data/train/utt2spk > data/train/spk2utt

    # Prepare scp files of noises and RIRs for training (used for on-the-fly mixing)
    # * The noise set is composed of two parts:
    #   (1) selected from MUSAN and Audioset (25390 samples, ~120 hours)
    #   (2) real meeting room noises recorded by high fidelity devices (98 clips, ~13 hours)
    #   NOTE: different noise data may have different sample rates.
    # * 28914 RIRs are simulated using the image method.
    sed -e 's/\.\(wav\|flac\)//' "${simu_data_path}/train_noise.lst" | \
        awk -F '/' '{print $NF}' > "${tmpdir}/utt_noise.list"
    paste -d' ' "${tmpdir}/utt_noise.list" "${simu_data_path}/train_noise.lst" > data/train/noises.scp
    find "${official_data_dir}/Training_set/train_record_noise/" -iname "*.wav" > "${tmpdir}/train_record_noise.list"
    sed -e 's/\.\(wav\|flac\)//' "${tmpdir}/train_record_noise.list" | \
        awk -F '/' '{print $NF}' > "${tmpdir}/utt_record_noise.list"
    paste -d' ' "${tmpdir}/utt_record_noise.list" "${tmpdir}/train_record_noise.list" >> data/train/noises.scp

    # NOTE: different RIRs may have different numbers of channels.
    cat "${simu_data_path}"/train_{circle,linear,non_uniform}_rir.lst > "${tmpdir}/train_rir.list"
    sed -e 's/\.wav//' "${tmpdir}/train_rir.list" | \
        awk -F '/' '{print $NF}' > "${tmpdir}/utt_rir.list"
    paste -d' ' "${tmpdir}/utt_rir.list" "${tmpdir}/train_rir.list" > data/train/rirs.scp

    utils/validate_data_dir.sh --no-feats --no-text data/train

    ####################
    # Development data #
    ####################
    mkdir -p data/dev
    cat "${simu_data_path}"/dev_{circle,linear,non_uniform}_simu_mix.config > ${tmpdir}/dev.config
    python local/prepare_dev_data.py \
        --audiodirs "${simu_data_path}/wavs/dev" \
        --outdir data/dev \
        ${tmpdir}/dev.config

    for f in noise1.scp spk1.scp utt2spk wav.scp; do
        mv data/dev/${f} data/dev/.${f}
        sort data/dev/.${f} > data/dev/${f}
        rm data/dev/.${f}
    done
    utils/utt2spk_to_spk2utt.pl data/dev/utt2spk > data/dev/spk2utt
    utils/validate_data_dir.sh --no-feats --no-text data/dev

    rm -rf "$tmpdir"
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
