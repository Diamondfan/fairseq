
#train

stage=0
end_stage=0

# Pretrain with Librispeech 960h
if [ $stage -le 0 ] && [ $end_stage -ge 0 ]; then
    this_dir=`pwd`
    exp_dir=$this_dir/exp_libri/test_lr1e-3/

    [ ! -d $exp_dir ] && mkdir -p $exp_dir
    train_log=train.log

    CUDA_VISIBLE_DEVICES="2,3" fairseq-hydra-train \
      --config-dir $this_dir/config/pretraining/ \
      --config-name wav2vec2_base_librispeech_my.yaml \
      task.data=$this_dir/libri_data/ \
      hydra.run.dir=$exp_dir \
      common.log_file=$train_log \
      distributed_training.distributed_world_size=2 \
      optimization.update_freq='[32]' \
      optimization.lr='[0.001]' \
      common.log_interval=100 > $exp_dir/log 2>&1 &
fi

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
    #model=${exp_dir}/checkpoints/checkpoint_best.pt \
    exp_dir=$this_dir/wav2vec_base/
    model=$this_dir/wav2vec_base/wav2vec_small_960h.pt
    subset="dev test" 
 
    for x in ${subset}; do
        append=
        decode_save_dir=${exp_dir}/results/decode_ogi_${x}${append}
        logfile=decode_${x}${append}
        CUDA_VISIBLE_DEVICES='3' python ../speech_recognition/new/infer.py \
            --config-dir $this_dir/config/decode \
            --config-name infer_viterbi \
            task.data=$this_dir/ogi_data \
            task.labels="ltr2" \
            task.normalize=false \
            common_eval.path=$model \
            common_eval.results_path=${decode_save_dir} \
            dataset.gen_subset=$x \
            common.log_file=${logfile}
    done
fi
