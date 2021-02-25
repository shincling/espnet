#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# run local/data.sh for more information
official_data_dir=
sample_rate=16k

train_set=train
valid_set=dev
test_sets="dev"

./enh.sh \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --audio_format wav \
    --fs ${sample_rate} \
    --ngpu 1 \
    --spk_num 1 \
    --local_data_opts "--stage 1 --stop-stage 3 --official_data_dir ${official_data_dir}" \
    --extra_wav_list "rirs.scp noises.scp" \
    --enh_args "--preprocessor_type conferencingspeech --rir_scp dump/raw/${train_set}/rirs.scp --rir_max_channel 8 --rir_apply_prob 1.0 --noise_scp dump/raw/${train_set}/noises.scp --noise_max_channel 1 --noise_apply_prob 1.0 --noise_db_range 0_30" \
    --enh_config conf/tuning/train_enh_beamformer_mvdr.yaml \
    --use_dereverb_ref false \
    --use_noise_ref false \
    --inference_model "valid.loss.best.pth" \
    "$@"
