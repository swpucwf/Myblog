#!/usr/bin/env bash

. ./path.sh || exit 1;

# machines configuration
CUDA_VISIBLE_DEVICES="0"
gpu_num=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
count=1

# general configuration
dump_cmd=utils/run.pl
nj=64

# feature configuration
data_dir="./data"
simu_feats_dir=$data_dir/ark_data/dump/simu_data/data
simu_feats_dir_chunk2000=$data_dir/ark_data/dump/simu_data_chunk2000/data
callhome_feats_dir_chunk2000=$data_dir/ark_data/dump/callhome_chunk2000/data
simu_train_dataset=train
simu_valid_dataset=dev
callhome_train_dataset=callhome1_spkall
callhome_valid_dataset=callhome2_spkall

# model average
simu_average_2spkr_start=91
simu_average_2spkr_end=100
simu_average_allspkr_start=16
simu_average_allspkr_end=25
callhome_average_start=91
callhome_average_end=100

exp_dir="."
input_size=345
stage=1
stop_stage=5

# exp tag
tag="exp1"

. local/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

simu_2spkr_diar_config=conf/train_diar_eend_ola_simu_2spkr.yaml
simu_allspkr_diar_config=conf/train_diar_eend_ola_simu_allspkr.yaml
simu_allspkr_chunk2000_diar_config=conf/train_diar_eend_ola_simu_allspkr_chunk2000.yaml
callhome_diar_config=conf/train_diar_eend_ola_callhome_chunk2000.yaml
simu_2spkr_model_dir="baseline_$(basename "${simu_2spkr_diar_config}" .yaml)_${tag}"
simu_allspkr_model_dir="baseline_$(basename "${simu_allspkr_diar_config}" .yaml)_${tag}"
simu_allspkr_chunk2000_model_dir="baseline_$(basename "${simu_allspkr_chunk2000_diar_config}" .yaml)_${tag}"
callhome_model_dir="baseline_$(basename "${callhome_diar_config}" .yaml)_${tag}"

# simulate mixture data for training and inference
if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "stage -1: Simulate mixture data for training and inference"
    echo "The detail can be found in https://github.com/hitachi-speech/EEND"
    echo "Before running this step, you should download and compile kaldi and set KALDI_ROOT in this script and path.sh"
    echo "This stage may take a long time, please waiting..."
    KALDI_ROOT=
    ln -s $KALDI_ROOT/egs/wsj/s5/steps steps
    ln -s $KALDI_ROOT/egs/wsj/s5/utils utils
    local/run_prepare_shared_eda.sh
fi

# Prepare data for training and inference
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Prepare data for training and inference"
    simu_opts_num_speaker_array=(1 2 3 4)
    simu_opts_sil_scale_array=(2 2 5 9)
    simu_opts_num_train=100000

    # for simulated data of chunk500 and chunk2000
    for dset in swb_sre_cv swb_sre_tr; do
        if [ "$dset" == "swb_sre_tr" ]; then
            n_mixtures=${simu_opts_num_train}
            dataset=train
        else
            n_mixtures=500
            dataset=dev
        fi
        simu_data_dir=${dset}_ns"$(IFS="n"; echo "${simu_opts_num_speaker_array[*]}")"_beta"$(IFS="n"; echo "${simu_opts_sil_scale_array[*]}")"_${n_mixtures}
        mkdir -p ${data_dir}/simu/data/${simu_data_dir}/.work
        split_scps=
        for n in $(seq $nj); do
            split_scps="$split_scps ${data_dir}/simu/data/${simu_data_dir}/.work/wav.scp.$n"
        done
        utils/split_scp.pl "${data_dir}/simu/data/${simu_data_dir}/wav.scp" $split_scps || exit 1
        python local/split.py ${data_dir}/simu/data/${simu_data_dir}
        # for chunk_size=500
        output_dir=${data_dir}/ark_data/dump/simu_data/$dataset
        mkdir -p $output_dir/.logs
        $dump_cmd --max-jobs-run $nj JOB=1:$nj $output_dir/.logs/dump.JOB.log \
        python local/dump_feature.py \
              --data_dir ${data_dir}/simu/data/${simu_data_dir}/.work \
              --output_dir $output_dir \
              --index JOB
        mkdir -p ${data_dir}/ark_data/dump/simu_data/data/$dataset
        cat ${data_dir}/ark_data/dump/simu_data/$dataset/feature.scp.* > ${data_dir}/ark_data/dump/simu_data/data/$dataset/feature.scp
        cat ${data_dir}/ark_data/dump/simu_data/$dataset/label.scp.* > ${data_dir}/ark_data/dump/simu_data/data/$dataset/label.scp
        paste -d" " ${data_dir}/ark_data/dump/simu_data/data/$dataset/feature.scp <(cut -f2 -d" " ${data_dir}/ark_data/dump/simu_data/data/$dataset/label.scp) > ${data_dir}/ark_data/dump/simu_data/data/$dataset/feats.scp
        grep "ns2" ${data_dir}/ark_data/dump/simu_data/data/$dataset/feats.scp > ${data_dir}/ark_data/dump/simu_data/data/$dataset/feats_2spkr.scp
        # for chunk_size=2000
        output_dir=${data_dir}/ark_data/dump/simu_data_chunk2000/$dataset
        mkdir -p $output_dir/.logs
        $dump_cmd --max-jobs-run $nj JOB=1:$nj $output_dir/.logs/dump.JOB.log \
        python local/dump_feature.py \
              --data_dir ${data_dir}/simu/data/${simu_data_dir}/.work \
              --output_dir $output_dir \
              --index JOB \
              --num_frames 2000
        mkdir -p ${data_dir}/ark_data/dump/simu_data_chunk2000/data/$dataset
        cat ${data_dir}/ark_data/dump/simu_data_chunk2000/$dataset/feature.scp.* > ${data_dir}/ark_data/dump/simu_data_chunk2000/data/$dataset/feature.scp
        cat ${data_dir}/ark_data/dump/simu_data_chunk2000/$dataset/label.scp.* > ${data_dir}/ark_data/dump/simu_data_chunk2000/data/$dataset/label.scp
        paste -d" " ${data_dir}/ark_data/dump/simu_data_chunk2000/data/$dataset/feature.scp <(cut -f2 -d" " ${data_dir}/ark_data/dump/simu_data_chunk2000/data/$dataset/label.scp) > ${data_dir}/ark_data/dump/simu_data_chunk2000/data/$dataset/feats.scp
    done

    # for callhome data
    for dset in callhome1_spkall callhome2_spkall; do
        find  $data_dir/eval/$dset  -maxdepth 1 -type f -exec cp {} {}.1 \;
        output_dir=${data_dir}/ark_data/dump/callhome_chunk2000/$dset
        mkdir -p $output_dir
        python local/dump_feature.py \
              --data_dir $data_dir/eval/$dset \
              --output_dir $output_dir \
              --index 1 \
              --num_frames 2000
        mkdir -p ${data_dir}/ark_data/dump/callhome_chunk2000/data/$dset
        paste -d" " ${data_dir}/ark_data/dump/callhome_chunk2000/$dset/feature.scp.1 <(cut -f2 -d" " ${data_dir}/ark_data/dump/callhome_chunk2000/$dset/label.scp.1) > ${data_dir}/ark_data/dump/callhome_chunk2000/data/$dset/feats.scp
    done
fi

# Training on simulated two-speaker data
world_size=$gpu_num
simu_2spkr_ave_id=avg${simu_average_2spkr_start}-${simu_average_2spkr_end}
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: Training on simulated two-speaker data"
    mkdir -p ${exp_dir}/exp/${simu_2spkr_model_dir}
    mkdir -p ${exp_dir}/exp/${simu_2spkr_model_dir}/log
    INIT_FILE=${exp_dir}/exp/${simu_2spkr_model_dir}/ddp_init
    if [ -f $INIT_FILE ];then
        rm -f $INIT_FILE
    fi
    init_method=file://$(readlink -f $INIT_FILE)
    echo "$0: init method is $init_method"
    for ((i = 0; i < $gpu_num; ++i)); do
        {
            rank=$i
            local_rank=$i
            gpu_id=$(echo $CUDA_VISIBLE_DEVICES | cut -d',' -f$[$i+1])
            train.py \
                --task_name diar \
                --gpu_id $gpu_id \
                --use_preprocessor false \
                --input_size $input_size \
                --data_dir ${simu_feats_dir} \
                --train_set ${simu_train_dataset} \
                --valid_set ${simu_valid_dataset} \
                --data_file_names "feats_2spkr.scp" \
                --resume true \
                --output_dir ${exp_dir}/exp/${simu_2spkr_model_dir} \
                --config $simu_2spkr_diar_config \
                --ngpu $gpu_num \
                --num_worker_count $count \
                --dist_init_method $init_method \
                --dist_world_size $world_size \
                --dist_rank $rank \
                --local_rank $local_rank 1> ${exp_dir}/exp/${simu_2spkr_model_dir}/log/train.log.$i 2>&1
        } &
        done
        wait
    echo "averaging model parameters into ${exp_dir}/exp/$simu_2spkr_model_dir/$simu_2spkr_ave_id.pb"
    models=`eval echo ${exp_dir}/exp/${simu_2spkr_model_dir}/{$simu_average_2spkr_start..$simu_average_2spkr_end}epoch.pb`
    python local/model_averaging.py ${exp_dir}/exp/${simu_2spkr_model_dir}/$simu_2spkr_ave_id.pb $models
fi

# Training on simulated all-speaker data
world_size=$gpu_num
simu_allspkr_ave_id=avg${simu_average_allspkr_start}-${simu_average_allspkr_end}
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: Training on simulated all-speaker data"
    mkdir -p ${exp_dir}/exp/${simu_allspkr_model_dir}
    mkdir -p ${exp_dir}/exp/${simu_allspkr_model_dir}/log
    INIT_FILE=${exp_dir}/exp/${simu_allspkr_model_dir}/ddp_init
    if [ -f $INIT_FILE ];then
        rm -f $INIT_FILE
    fi
    init_method=file://$(readlink -f $INIT_FILE)
    echo "$0: init method is $init_method"
    for ((i = 0; i < $gpu_num; ++i)); do
        {
            rank=$i
            local_rank=$i
            gpu_id=$(echo $CUDA_VISIBLE_DEVICES | cut -d',' -f$[$i+1])
            train.py \
                --task_name diar \
                --gpu_id $gpu_id \
                --use_preprocessor false \
                --input_size $input_size \
                --data_dir ${simu_feats_dir} \
                --train_set ${simu_train_dataset} \
                --valid_set ${simu_valid_dataset} \
                --data_file_names "feats.scp" \
                --resume true \
                --init_param ${exp_dir}/exp/${simu_2spkr_model_dir}/$simu_2spkr_ave_id.pb \
                --output_dir ${exp_dir}/exp/${simu_allspkr_model_dir} \
                --config $simu_allspkr_diar_config \
                --ngpu $gpu_num \
                --num_worker_count $count \
                --dist_init_method $init_method \
                --dist_world_size $world_size \
                --dist_rank $rank \
                --local_rank $local_rank 1> ${exp_dir}/exp/${simu_allspkr_model_dir}/log/train.log.$i 2>&1
        } &
        done
        wait
    echo "averaging model parameters into ${exp_dir}/exp/$simu_allspkr_model_dir/$simu_allspkr_ave_id.pb"
    models=`eval echo ${exp_dir}/exp/${simu_allspkr_model_dir}/{$simu_average_allspkr_start..$simu_average_allspkr_end}epoch.pb`
    python local/model_averaging.py ${exp_dir}/exp/${simu_allspkr_model_dir}/$simu_allspkr_ave_id.pb $models
fi

# Training on simulated all-speaker data with chunk_size 2000
world_size=$gpu_num
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: Training on simulated all-speaker data with chunk_size 2000"
    mkdir -p ${exp_dir}/exp/${simu_allspkr_chunk2000_model_dir}
    mkdir -p ${exp_dir}/exp/${simu_allspkr_chunk2000_model_dir}/log
    INIT_FILE=${exp_dir}/exp/${simu_allspkr_chunk2000_model_dir}/ddp_init
    if [ -f $INIT_FILE ];then
        rm -f $INIT_FILE
    fi
    init_method=file://$(readlink -f $INIT_FILE)
    echo "$0: init method is $init_method"
    for ((i = 0; i < $gpu_num; ++i)); do
        {
            rank=$i
            local_rank=$i
            gpu_id=$(echo $CUDA_VISIBLE_DEVICES | cut -d',' -f$[$i+1])
            train.py \
                --task_name diar \
                --gpu_id $gpu_id \
                --use_preprocessor false \
                --input_size $input_size \
                --data_dir ${simu_feats_dir_chunk2000} \
                --train_set ${simu_train_dataset} \
                --valid_set ${simu_valid_dataset} \
                --data_file_names "feats.scp" \
                --resume true \
                --init_param ${exp_dir}/exp/${simu_allspkr_model_dir}/$simu_allspkr_ave_id.pb \
                --output_dir ${exp_dir}/exp/${simu_allspkr_chunk2000_model_dir} \
                --config $simu_allspkr_chunk2000_diar_config \
                --ngpu $gpu_num \
                --num_worker_count $count \
                --dist_init_method $init_method \
                --dist_world_size $world_size \
                --dist_rank $rank \
                --local_rank $local_rank 1> ${exp_dir}/exp/${simu_allspkr_chunk2000_model_dir}/log/train.log.$i 2>&1
        } &
        done
        wait
fi

# Training on callhome all-speaker data with chunk_size 2000
world_size=$gpu_num
callhome_ave_id=avg${callhome_average_start}-${callhome_average_end}
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Training on callhome all-speaker data with chunk_size 2000"
    mkdir -p ${exp_dir}/exp/${callhome_model_dir}
    mkdir -p ${exp_dir}/exp/${callhome_model_dir}/log
    INIT_FILE=${exp_dir}/exp/${callhome_model_dir}/ddp_init
    if [ -f $INIT_FILE ];then
        rm -f $INIT_FILE
    fi
    init_method=file://$(readlink -f $INIT_FILE)
    echo "$0: init method is $init_method"
    for ((i = 0; i < $gpu_num; ++i)); do
        {
            rank=$i
            local_rank=$i
            gpu_id=$(echo $CUDA_VISIBLE_DEVICES | cut -d',' -f$[$i+1])
            train.py \
                --task_name diar \
                --gpu_id $gpu_id \
                --use_preprocessor false \
                --input_size $input_size \
                --data_dir ${callhome_feats_dir_chunk2000} \
                --train_set ${callhome_train_dataset} \
                --valid_set ${callhome_valid_dataset} \
                --data_file_names "feats.scp" \
                --resume true \
                --init_param ${exp_dir}/exp/${simu_allspkr_chunk2000_model_dir}/1epoch.pb \
                --output_dir ${exp_dir}/exp/${callhome_model_dir} \
                --config $callhome_diar_config \
                --ngpu $gpu_num \
                --num_worker_count $count \
                --dist_init_method $init_method \
                --dist_world_size $world_size \
                --dist_rank $rank \
                --local_rank $local_rank 1> ${exp_dir}/exp/${callhome_model_dir}/log/train.log.$i 2>&1
        } &
        done
        wait
    echo "averaging model parameters into ${exp_dir}/exp/$callhome_model_dir/$callhome_ave_id.pb"
    models=`eval echo ${exp_dir}/exp/${callhome_model_dir}/{$callhome_average_start..$callhome_average_end}epoch.pb`
    python local/model_averaging.py ${exp_dir}/exp/${callhome_model_dir}/$callhome_ave_id.pb $models
fi

# inference and compute DER
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "Inference"
    mkdir -p ${exp_dir}/exp/${callhome_model_dir}/inference/log
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python local/infer.py \
        --config_file ${exp_dir}/exp/${callhome_model_dir}/config.yaml \
        --model_file ${exp_dir}/exp/${callhome_model_dir}/$callhome_ave_id.pb \
        --output_rttm_file ${exp_dir}/exp/${callhome_model_dir}/inference/rttm \
        --wav_scp_file $data_dir/eval/callhome2_spkall/wav.scp \
        1> ${exp_dir}/exp/${callhome_model_dir}/inference/log/infer.log 2>&1
    md-eval.pl -c 0.25 \
          -r ${data_dir}/eval/${callhome_valid_dataset}/rttm \
          -s ${exp_dir}/exp/${callhome_model_dir}/inference/rttm > ${exp_dir}/exp/${callhome_model_dir}/inference/result_med11_collar0.25 2>/dev/null || exit
fi