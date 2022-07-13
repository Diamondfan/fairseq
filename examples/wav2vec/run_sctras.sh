
#train

stage=0
end_stage=0

kaldi_data_style_dir=/data/ruchao/workdir/SSLASR/egs/OGI/data/

# Pretrain with OGI
if [ $stage -le 0 ] && [ $end_stage -ge 0 ]; then
    wav2vec_base_model=wav2vec_base/wav2vec_small.pt
    this_dir=`pwd`
    data_dir=$this_dir/ogi_data/
    exp_dir=exp_ogi/test/

    [ ! -d $exp_dir ] && mkdir -p $exp_dir
    train_log=train.log

    CUDA_VISIBLE_DEVICES="2" fairseq-hydra-train \
      --config-dir $this_dir/config/pretraining/ \
      --config-name ogi_sp.yaml \
      task.data=$this_dir/ogi_data/ \
      hydra.run.dir=${exp_dir} \
      common.log_file=${train_log} \
      model.bottleneck_dim=0 \
      model.freeze_adapter=false \
      model.freeze_backbone=false \
      model.no_pretrained_weights=false \
      model.pretrained_weights_path=$this_dir/$wav2vec_base_model \
      model.use_first_adapter=false \
      model.adapter_before_quant=false
fi

# finetuning stage
if [ $stage -le 1 ] && [ $end_stage -ge 1 ]; then
  wav2vec_base_model=exp_ogi/exp_pt_on_ogi_bn1024_adapter_update_backbone_freeze_adapterbquant/exp_ogi_sp_s200k_lr5e-5_wd01/checkpoints/checkpoint_best.pt

  this_dir=`pwd`
  data_dir=$this_dir/ogi_data/
  exp_dir=exp_ogi/ft_wav2vec_wadapter_before_quant_bn1024_lr5e-5_bs16_update4_steps40k_lr3e-5_fix_adapter/
  
  [ ! -d $exp_dir ] && mkdir -p $exp_dir
  train_log=train.log

  # set log dir:        hydra.run.dir
  # set log file name:  common.log_file
  #export HYDRA_FULL_ERROR=1 

  CUDA_VISIBLE_DEVICES="0" fairseq-hydra-train \
    --config-dir ${this_dir}/config/finetuning \
    --config-name ogi_sp.yaml \
    task.data=$data_dir \
    model.w2v_path=$this_dir/${wav2vec_base_model} \
    hydra.run.dir=${exp_dir} \
    common.log_file=${train_log} \
    model.bottleneck_dim=1024 \
    model.freeze_adapter=true \
    model.freeze_backbone=false \
    model.no_pretrained_weights=false \
    model.use_first_adapter=true \
    model.adapter_before_quant=true > $exp_dir/log 2>&1 &
fi

if [ $stage -le 2 ] && [ $end_stage -ge 2 ]; then
    this_dir=`pwd`
    exp_dir=$this_dir/exp_ogi/ft_wav2vec_wadapter_before_quant_bn1024_lr5e-5_bs16_update4_steps40k_lr3e-5_fix_adapter/

    model=${exp_dir}/checkpoints/checkpoint_best.pt \
    subset="dev test" 
 
    for x in ${subset}; do
        append=
        decode_save_dir=${exp_dir}/results/decode_${x}${append}
        logfile=decode_${x}${append}
        CUDA_VISIBLE_DEVICES='0' python ../speech_recognition/new/infer.py \
            --config-dir $this_dir/config/decode \
            --config-name infer_viterbi \
            task.data=$this_dir/ogi_data \
            task.normalize=false \
            common_eval.path=$model \
            common_eval.results_path=${decode_save_dir} \
            dataset.gen_subset=$x \
            common.log_file=${logfile}
    done
fi
