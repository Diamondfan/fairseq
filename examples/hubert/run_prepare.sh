
stage=5
end_stage=5

kaldi_data_style_dir=/data/ruchao/workdir/SSLASR/egs/OGI/data/

if [ $stage -le 1 ] && [ $end_stage -ge 1 ]; then
  echo "prepare wav.tsv!"
  [ ! -d ogi_data ] && mkdir -p ogi_data
  wav_scp=$kaldi_data_style_dir/train_sp/wav.scp
  wav_dir=ogi_data/wav/
  [ ! -d $wav_dir ] && mkdir -p $wav_dir
  cat $wav_scp | awk -v wav_dir=$wav_dir -F' ' '{for(i=2;i<NF;i++) printf($i" ")};{print "> "wav_dir"/"$1".wav"}' > convert.sh  
  bash convert.sh
  rm -f convert.sh
  
  train_scp=ogi_data/train_sp.tsv
  touch $train_scp
  echo `pwd` > $train_scp
  find $wav_dir -iname "*.wav" | sort -k1,1 | awk '{print $1"\t1" }' >> $train_scp
  
  #dev and test test
  for tset in dev test; do
    wav_scp=$kaldi_data_style_dir/$tset/wav.scp
    tsv_scp=ogi_data/${tset}.tsv
    touch $tsv_scp
    echo `pwd` > $tsv_scp
    cat $wav_scp | awk -F' ' '{print $2"\t1"}' >> ogi_data/${tset}.tsv
  done
fi

if [ $stage -le 2 ] && [ $end_stage -ge 2 ]; then
  echo "prepare label for CTC finetuning"
  
  for tset in train_sp dev test; do
    text_scp=$kaldi_data_style_dir/$tset/token_char.scp
    cat $text_scp | cut -d" " -f2- > ogi_data/$tset.ltr
  done

  cat $kaldi_data_style_dir/dict/vocab_char.txt | awk '{print $1" 1" }' > ogi_data/dict.ltr.txt
fi
