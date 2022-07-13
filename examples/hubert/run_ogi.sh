
#train
stage=3
end_stage=3

# example for hubert
if [ $stage -le 1 ] && [ $end_stage -ge 1 ]; then
  echo "pretrain with OGI"
  #exp_dir=debug #
  #exp_dir=jinhan/pre_ogi/exp_ogi_sp_lr1e-4_200k_wrm32k_bsz16_update_1_cor_num_sample
  #exp_dir=jinhan/pre_ogi_bn1024_adapter_update_backbone_freeze/exp_ogi_sp_lr5e-4_100k_wrm32k_bsz4_update_4_cor_num_sample_wd01
  #exp_dir=debug2/ #jinhan/hubert_6_all_data_pre_ogi_bn1024_adapter_update_backbone_freeze_adapterbquant/exp_ogi_sp_lr1e-4_100k_wrm16k_bsz4_update_4_cor_num_sample_wd01
  exp_dir=jinhan/pre_ogi/exp_ogi_sp_lr5e-4_100k_wrm8k_bsz4_update_4_cor_num_sample_wd01
  hubert_base_model=hubert_base/hubert_base_ls960.pt  
  #hubert_base_model=ogi_pretrain/ogi_pretrain.pt
  this_dir=`pwd`
  train_log=train.log
  lab_dir=$this_dir/lab_dir_hubert_6 #label_dir_mfcc
  echo $this_dir

  # set log dir:        hydra.run.dir
  # set log file name:  common.log_file
  CUDA_VISIBLE_DEVICES="0" python ../../fairseq_cli/hydra_train.py \
      --config-dir $this_dir/config/pretrain/ \
      --config-name ogi_sp \
      task.data=$this_dir/ogi_data/ \
      task.label_dir=$lab_dir \
      task.labels='[km]' \
      model.label_rate=50 \
      model.w2v_path=$this_dir/$hubert_base_model \
      hydra.run.dir=${exp_dir} \
      common.log_file=${train_log} \
      model.bottleneck_dim=0 \
      model.freeze_adapter=false \
      model.freeze_backbone=false \
      model.no_pretrained_weights=false \
      model.pretrained_weights_path=$this_dir/$hubert_base_model \
      model.use_first_adapter=false \
      model.adapter_before_quant=false >> train_pre_ogi_5e-4_100k_wrm8k.log
fi
  
 
if [ $stage -le 2 ] && [ $end_stage -ge 2 ]; then
  echo "Finetune with OGI"
  #hubert_base_model=hubert_base/hubert_base_ls960.pt  
  #hubert_base_model=exp_myst/pt_segment_woadapter_bs16_updates4_max800k_min0p5s_steps200k_lr1e-4/checkpoint_best.pt
  #hubert_base_model=exp_myst/pt_segment_wadapter_bn1024_freeze_backbone_before_quant_bs16_updates4_min0p5s_steps200k_lr5e-4/checkpoint_best.pt
  #hubert_base_model=exp_ogi/pre_ogi/exp_ogi_sp_lr5e-4_100k_wrm8k_bsz4_update_4_cor_num_sample_wd01/checkpoints/checkpoint_best.pt
  hubert_base_model=exp_ogi/pre_ogi_bn1024_adapter_update_backbone_freeze_adpterbquant_unfreeze_emb/exp_ogi_sp_lr5e-4_100k_wrm8k_bsz4_update_4_cor_num_sample_wd01/checkpoints/checkpoint_best.pt
  
  this_dir=`pwd`
  data_dir=$this_dir/ogi_data

  #exp_dir=exp_ogi/ft_hubert_frmyst_woadapter_lr1e-4_bs16_update4_steps40k_lr7e-5/
  #exp_dir=exp_ogi/ft_hubert_frmyst_wadapter_bn1024_before_quant_bs16_update4_steps40k_lr7e-5/
  #exp_dir=exp_ogi/ft_hubert_base_bs16_update4_steps40k_lr1e-4_adapter_finetuning_bn1024_freeze_backbone/
  #exp_dir=exp_ogi/ft_hubert_base_bs16_update4_steps100k_lr8e-4_noamwarm10k_lstm_generator_d1024_n2_drop02_freeze_backbone/
  #exp_dir=exp_ogi/ft_hubert_woadapter_lr1e-4_bs16_update4_steps100k_lr8e-4_noamwarm10k_lstm_generator_d1024_n2_drop02_freeze_backbone/
  #exp_dir=exp_ogi/ft_hubert_wadapter_before_quant_bn1024_lr5e-4_bs16_update4_steps100k_lr8e-4_noamwarm10k_lstm_generator_d1024_n2_drop02_freeze_backbone/
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
    #exp_dir=$this_dir/exp_ogi/ft_hubert_frmyst_woadapter_lr1e-4_bs16_update4_steps40k_lr7e-5/
    #exp_dir=$this_dir/exp_ogi/ft_hubert_frmyst_wadapter_bn1024_before_quant_bs16_update4_steps40k_lr7e-5/
    #exp_dir=$this_dir/exp_ogi/ft_hubert_base_bs16_update4_steps100k_lr8e-4_noamwarm10k_lstm_generator_d1024_n2_drop02_freeze_backbone/
    #exp_dir=$this_dir/exp_ogi/ft_hubert_woadapter_lr1e-4_bs16_update4_steps100k_lr8e-4_noamwarm10k_lstm_generator_d1024_n2_drop02_freeze_backbone/
    #exp_dir=$this_dir/exp_ogi/ft_hubert_wadapter_before_quant_bn1024_lr5e-4_bs16_update4_steps100k_lr8e-4_noamwarm10k_lstm_generator_d1024_n2_drop02_freeze_backbone/
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
