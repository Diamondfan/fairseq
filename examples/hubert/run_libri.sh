# finetuning on librispeech 100h data

stage=1
end_stage=3


# finetuning stage
if [ $stage -le 1 ] && [ $end_stage -ge 1 ]; then
  hubert_base_model=pretrained_models/hubert_base_ls960.pt
  
  this_dir=`pwd`
  data_dir=$this_dir/libri_100h/
  exp_dir=exp_libri100/ft_hubert_from_libri960_80k_lr3e-5_stdltr/
 
  [ ! -d $exp_dir ] && mkdir -p $exp_dir
  train_log=train.log

  CUDA_VISIBLE_DEVICES="0,1" python ../../fairseq_cli/hydra_train.py \
    --config-dir ${this_dir}/config/finetune \
    --config-name base_100h.yaml \
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
    model.adapter_before_quant=false > $exp_dir/log 2>&1 #&
fi

if [ $stage -le 2 ] && [ $end_stage -ge 2 ]; then
    this_dir=`pwd`
    exp_dir=$this_dir/exp_libri100/ft_hubert_from_libri960_80k_lr3e-5_stdltr/

    model=$exp_dir/checkpoints/checkpoint_last.pt \
    subset="test_clean test_other dev_clean dev_other"
    data_dir=libri_100h  #myst_data

    for x in ${subset}; do
        append="_last"
        decode_save_dir=${exp_dir}/results/decode_${x}${append}
        logfile=decode_${x}${append}
        CUDA_VISIBLE_DEVICES="0,1" python ../speech_recognition/new/infer.py \
            --config-dir $this_dir/config/decode \
            --config-name infer_viterbi \
            task.data=$this_dir/$data_dir/ \
            task.normalize=false \
            common_eval.path=$model \
            common_eval.results_path=${decode_save_dir} \
            dataset.gen_subset=$x \
            common.log_file=${logfile}
    done
fi

if [ $stage -le 3 ] && [ $end_stage -ge 3 ]; then
    this_dir=`pwd`
    exp_dir=$this_dir/exp_libri100/ft_hubert_from_libri960_80k_lr3e-5_stdltr/
    
    model=$exp_dir/checkpoints/checkpoint_last.pt \
    subset="test_clean test_other dev_clean dev_other"
    data_dir=libri_100h #myst_data

    for x in ${subset}; do
        append="_last"
        decode_save_dir=${exp_dir}/results/decode_${x}${append}
        logfile=decode_${x}${append}
        CUDA_VISIBLE_DEVICES="0,1" python ../speech_recognition/new/infer.py \
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


