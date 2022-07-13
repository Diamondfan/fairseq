
#train

stage=3
end_stage=3

kaldi_data_style_dir=/data/ruchao/workdir/SSLASR/egs/MyST/data/

# Adapt
if [ $stage -le 1 ] && [ $end_stage -ge 1 ]; then
    wav2vec_base_model=wav2vec_base/wav2vec_small.pt
    this_dir=`pwd`
    data_dir=$this_dir/myst_data/
    exp_dir=exp/pt_wadapter_bn1024_freeze_backbone_before_quant_steps200k_lr5e-4_warmpup32k/
    [ ! -d $exp_dir ] && mkdir -p $exp_dir
    train_log=train.log

    srun fairseq-hydra-train \
      --config-dir $this_dir/config/pretraining/ \
      --config-name myst_sp.yaml \
      task.data=$data_dir \
      hydra.run.dir=${exp_dir} \
      common.log_file=${train_log} \
      model.bottleneck_dim=1024 \
      model.freeze_adapter=false \
      model.freeze_backbone=true \
      model.no_pretrained_weights=false \
      model.pretrained_weights_path=$this_dir/$wav2vec_base_model \
      model.use_first_adapter=true \
      model.adapter_before_quant=true > $exp_dir/log 2>&1
fi

# finetuning stage
if [ $stage -le 2 ] && [ $end_stage -ge 2 ]; then
  wav2vec_base_model=wav2vec_base/wav2vec_small.pt 
  #wav2vec_base_model=exp/pt_woadapter_steps200k_lr1e-4/checkpoints/checkpoint_best.pt
  #wav2vec_base_model=exp/pt_wadapter_bn1024_freeze_backbone_before_quant_steps200k_lr5e-4_warmpup32k/checkpoints/checkpoint_best.pt
  
  this_dir=`pwd`
  data_dir=$this_dir/myst_data/

  #exp_dir=exp/ft_wav2vec_base_wadapter_bn1024_before_quant_bs16_steps130k_lr1e-4_warmup32k_fpfalse/
  exp_dir=exp_myst/ft_wav2vec_base_bs16_update7_max50s_min0p5s_steps100k_lr5e-4_lstm1024_nlayer2_drp02_freeze_backbone/
  [ ! -d $exp_dir ] && mkdir -p $exp_dir
  train_log=train.log

  CUDA_VISIBLE_DEVICES="3" fairseq-hydra-train \
    --config-dir ${this_dir}/config/finetuning \
    --config-name myst_sp.yaml \
    task.data=$data_dir \
    model.w2v_path=$this_dir/${wav2vec_base_model} \
    hydra.run.dir=${exp_dir} \
    common.log_file=${train_log} \
    model.bottleneck_dim=0 \
    model.freeze_adapter=true \
    model.freeze_backbone=true \
    model.no_pretrained_weights=false \
    model.use_first_adapter=false \
    model.adapter_before_quant=false > $exp_dir/log 2>&1 &
fi

if [ $stage -le 3 ] && [ $end_stage -ge 3 ]; then
    this_dir=`pwd`
    #exp_dir=$this_dir/exp_myst/ft_wav2vec_base_bs16_update7_max50s_min0p5s_1gpu_steps100k_lr3e-5/
    #model=$exp_dir/checkpoints/checkpoint_best.pt \

    exp_dir=$this_dir/wav2vec_base/
    model=$this_dir/wav2vec_base/wav2vec_small_960h.pt

    subset="development test"

    for x in ${subset}; do
        append=
        decode_save_dir=${exp_dir}/results/decode_myst_${x}${append}
        logfile=decode_${x}${append}
        CUDA_VISIBLE_DEVICES='3' python ../speech_recognition/new/infer.py \
            --config-dir $this_dir/config/decode \
            --config-name infer_viterbi \
            task.data=$this_dir/myst_data \
            task.labels='ltr2' \
            task.normalize=false \
            common_eval.path=$model \
            common_eval.results_path=${decode_save_dir} \
            dataset.gen_subset=$x \
            common.log_file=${logfile}
    done
fi

