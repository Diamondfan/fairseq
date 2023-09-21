# 2022, 2023 SPAPL
# Ruchao Fan
# DRAFT training and testing for MyST data

stage=1
end_stage=4

kaldi_data_style_dir=/data/ruchao/workdir/cassnat_asr/egs/MyST/data/

# pretrain myst model with mfcc clustered ids (first iteration)
if [ $stage -le -1 ] && [ $end_stage -ge -1 ]; then
    this_dir=`pwd`
    data_dir=$this_dir/myst_data/
    lab_dir=$this_dir/myst_mfcc_kmeans/myst_lab/
    exp_dir=exp_myst/hubert_iter1_mfcc_cluster100_lr4e-4/
    [ ! -d $exp_dir ] && mkdir -p $exp_dir
    train_log=train.log

    CUDA_VISIBLE_DEVICES="0,1,2,3" python ../../fairseq_cli/hydra_train.py \
      --config-dir $this_dir/config/pretrain/ \
      --config-name hubert_base_myst_all.yaml \
      task.data=$data_dir \
      task.label_dir=$lab_dir \
      task.labels='[km]' \
      model.label_rate=100 \
      hydra.run.dir=${exp_dir} \
      common.log_file=${train_log} \
      model.bottleneck_dim=0 \
      model.freeze_adapter=false \
      model.freeze_backbone=false \
      model.no_pretrained_weights=true \
      model.use_first_adapter=false \
      model.adapter_before_quant=false > $exp_dir/log 2>&1 &
fi

# pretrain myst model with hubert feature clustered ids (2nd iteration)
if [ $stage -le 0 ] && [ $end_stage -ge 0 ]; then
    this_dir=`pwd`
    data_dir=$this_dir/myst_data/
    lab_dir=$this_dir/myst_hubert_feat_kmeans/myst_lab/
    exp_dir=exp_myst/hubert_iter2_hubert_feat_cluster500_lr4e-4/
    [ ! -d $exp_dir ] && mkdir -p $exp_dir
    train_log=train.log

    CUDA_VISIBLE_DEVICES="0,1,2,3" python ../../fairseq_cli/hydra_train.py \
      --config-dir $this_dir/config/pretrain/ \
      --config-name hubert_base_myst_all.yaml \
      task.data=$data_dir \
      task.label_dir=$lab_dir \
      task.labels='[km]' \
      model.label_rate=50 \
      hydra.run.dir=${exp_dir} \
      common.log_file=${train_log} \
      model.bottleneck_dim=0 \
      model.freeze_adapter=false \
      model.freeze_backbone=false \
      model.no_pretrained_weights=true \
      model.use_first_adapter=false \
      model.adapter_before_quant=false > $exp_dir/log 2>&1 &
fi


# model fusion
if [ $stage -le 1 ] && [ $end_stage -ge 1 ]; then
    # run model fusion script here
    echo "a"
fi


# finetuning stage
if [ $stage -le 2 ] && [ $end_stage -ge 2 ]; then
  #hubert_base_model=pretrained_models/hubert_base_ls960.pt
  hubert_base_model=pretrained_models/hubert_large_ll60k.pt

  this_dir=`pwd`
  data_dir=$this_dir/myst_data/
  exp_dir=exp_myst/ft_hubert_from_librilarge_50s_max35s_160k_lr5e-5_stdltr/

  [ ! -d $exp_dir ] && mkdir -p $exp_dir
  train_log=train.log

  CUDA_VISIBLE_DEVICES="2,3" python ../../fairseq_cli/hydra_train.py \
    --config-dir ${this_dir}/config/finetune \
    --config-name large_myst_sp.yaml \
    task.data=$data_dir \
    task.label_dir=$data_dir \
    model.w2v_path=$this_dir/${hubert_base_model} \
    hydra.run.dir=${exp_dir} \
    common.log_file=${train_log} \
    model.bottleneck_dim=0 \
    model.freeze_adapter=false \
    model.freeze_backbone=false \
    model.no_pretrained_weights=false \
    model.use_first_adapter=false \
    model.adapter_before_quant=false >> $exp_dir/log 2>&1 &
fi

if [ $stage -le 3 ] && [ $end_stage -ge 3 ]; then
    this_dir=`pwd`
    #exp_dir=$this_dir/exp_myst/ft_hubert_from_libri960_160k_lr5e-5_stdltr/
    exp_dir=$this_dir/exp_myst/ft_hubert_from_librilarge_50s_max35s_160k_lr5e-5_stdltr/

    model=$exp_dir/checkpoints/checkpoint_last.pt \
    subset="dev_clean dev_other test_clean test_other" # "development test" #
    data_dir=$this_dir/libri_100h

    for x in ${subset}; do
        append="_last"
        decode_save_dir=${exp_dir}/results/decode_${x}${append}
        logfile=decode_${x}${append}
        CUDA_VISIBLE_DEVICES="2,3" python ../speech_recognition/new/infer.py \
            --config-dir $this_dir/config/decode \
            --config-name infer_viterbi \
            task.data=$data_dir \
            task.normalize=false \
            common_eval.path=$model \
            common_eval.results_path=${decode_save_dir} \
            dataset.gen_subset=$x \
            common.log_file=${logfile}
    done
fi

if [ $stage -le 4 ] && [ $end_stage -ge 4 ]; then
    this_dir=`pwd`
    #exp_dir=$this_dir/exp_myst/ft_hubert_from_libri960_160k_lr5e-5_stdltr/
    exp_dir=$this_dir/exp_myst/ft_hubert_from_librilarge_50s_max35s_160k_lr5e-5_stdltr/

    model=$exp_dir/checkpoints/checkpoint_last.pt \
    subset="test_clean test_other dev_clean dev_other"
    data_dir=libri_100h #myst_data

    for x in ${subset}; do
        append="_last"
        decode_save_dir=${exp_dir}/results/decode_${x}${append}
        logfile=decode_${x}${append}
        CUDA_VISIBLE_DEVICES="2,3" python ../speech_recognition/new/infer.py \
            --config-dir $this_dir/config/decode \
            --config-name infer_kenlm \
            task.data=$this_dir/$data_dir/ \
            task.normalize=false \
            common_eval.path=$model \
            common_eval.results_path=${decode_save_dir} \
            dataset.gen_subset=$x \
            common.log_file=${logfile} \
            decoding.lexicon=$this_dir/pretrained_models/librispeech-lexicon-char.txt \
            decoding.lmpath=$this_dir/pretrained_models/4-gram.bin
    done
fi

