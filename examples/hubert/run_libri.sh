
#train
stage=1
end_stage=1

# example for hubert
if [ $stage -le 1 ] && [ $end_stage -ge 1 ]; then
  echo "pretrain with OGI"
  
  this_dir=`pwd`
  train_log=train.log2
  exp_dir=$this_dir/exp_libri/test
  [ ! -d $exp_dir ] && mkdir -p $exp_dir
  lab_dir=$this_dir/hubert_kmeans_libri/lab_layer6/
  echo $this_dir

  # set log dir:        hydra.run.dir
  # set log file name:  common.log_file
  CUDA_VISIBLE_DEVICES="3" python ../../fairseq_cli/hydra_train.py \
      --config-dir $this_dir/config/pretrain/ \
      --config-name hubert_base_librispeech \
      task.data=$this_dir/libri_data/ \
      task.label_dir=$lab_dir \
      task.labels='[km]' \
      model.label_rate=50 \
      hydra.run.dir=${exp_dir} \
      common.log_file=${train_log} \
      distributed_training.distributed_world_size=1 \
      optimization.update_freq='[16]' #> $exp_dir/train.log 2>&1 &
fi
  
 
if [ $stage -le 2 ] && [ $end_stage -ge 2 ]; then
  echo "Finetune with OGI"
  hubert_base_model=exp_ogi/pre_ogi_bn1024_adapter_update_backbone_freeze_adpterbquant_unfreeze_emb/exp_ogi_sp_lr5e-4_100k_wrm8k_bsz4_update_4_cor_num_sample_wd01/checkpoints/checkpoint_best.pt
  
  this_dir=`pwd`
  data_dir=$this_dir/ogi_data

  exp_dir=exp_ogi/ft_hubert_wadapter_before_quant_bn1024_lr5e-4_bs16_update4_steps40k_lr7e-5_fix_adapter/  
  
  [ ! -d $exp_dir ] && mkdir -p $exp_dir

  # set log dir:        hydra.run.dir
  # set log file name:  common.log_file
  train_log=train.log

  CUDA_VISIBLE_DEVICES="1" fairseq-hydra-train \
    --config-dir config/finetune \
    --config-name ogi_sp.yaml \
    task.data=$data_dir \
    task.label_dir=$data_dir \
    model.w2v_path=$this_dir/$hubert_base_model \
    hydra.run.dir=$exp_dir \
    common.log_file=${train_log} \
    model.bottleneck_dim=1024 \
    model.freeze_adapter=true \
    model.freeze_backbone=false \
    model.no_pretrained_weights=false \
    model.use_first_adapter=true \
    model.adapter_before_quant=true > $exp_dir/log 2>&1 &
fi


if [ $stage -le 3 ] && [ $end_stage -ge 3 ]; then
    this_dir=`pwd`
    exp_dir=$this_dir/exp_ogi/ft_hubert_wadapter_before_quant_bn1024_lr5e-4_bs16_update4_steps40k_lr7e-5_fix_adapter/

    model=${exp_dir}/checkpoints/checkpoint_best.pt
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
