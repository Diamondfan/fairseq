
#train

stage=1
end_stage=1

kaldi_data_style_dir=/data/ruchao/workdir/cassnat_asr/egs/MyST/data/

if [ $stage -le 1 ] && [ $end_stage -ge 1 ]; then
  echo "prepare wav.tsv!"
  [ ! -d myst_data ] && mkdir -p myst_data
  for tset in train train_wotrn; do #development test train_sp; do
    echo $tset
    wav_scp=$kaldi_data_style_dir/$tset/wav.scp
    wav_dir=myst_data/wav/$tset/
    [ ! -d $wav_dir ] && mkdir -p $wav_dir
    cat $wav_scp | awk -v wav_dir=$wav_dir -F' ' '{for(i=2;i<NF;i++) printf($i" ")};{print "> "wav_dir"/"$1".wav"}' > convert.sh  
    bash convert.sh
    rm -f convert.sh
  
    tsv_scp=myst_data/${tset}.tsv
    touch $tsv_scp
    echo `pwd` > $tsv_scp
    find $wav_dir -iname "*.wav" | sort -k1,1 > a 
    python read_wavsize.py a >> $tsv_scp
    rm -f a
  done
fi

if [ $stage -le 2 ] && [ $end_stage -ge 2 ]; then
  echo "prepare label for CTC finetuning"
  
  for tset in train_sp development test; do
    echo "Processing $tset ...."
    text_scp=$kaldi_data_style_dir/$tset/token_char.scp
    cat $text_scp | cut -d" " -f2- > myst_data/$tset.ltr
  done

  #cat $kaldi_data_style_dir/dict/vocab_char.txt | awk '{print $1" 1" }' > myst_data/dict.ltr.txt
fi

