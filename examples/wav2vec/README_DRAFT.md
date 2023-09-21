# Domain Responsible Adaptation and Finetuning (DRAFT)

Code and scripts to run domain responsible adaptation and finetuning for Wav2vec2.0. The similar operations can be applied to HuBERT.

## The three-stage DRAFT training:
1. Pre-training with source domain data and the Wav2vec2.0 loss, which is typically opensourced and can be used directly. In our case, we use the Wav2vec2.0-Base model pretrained with Librispeech 960 hours data [download](https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt).
2. Domain responsible adaptation: Continual pre-training with target domain data and the Wav2vec2.0 loss. Residual adapters are added after each layer in encoder for efficient contiual learning while other modules of the encoder are freezed.
3. Domain responsible finetuning: Full finetuning of the encoder and residual adapters with the taget domain data and an ASR loss.

## Usage

DRAFT is built based on Fairseq repo and the examples in Wav2vec2.0 and HuBERT directories.

### 1. prepare the data
First, please follow the original README to prepare the data, including wav path and transcript for ASR finetuning. Example: train_clean_100.tsv (each line is the path of audio and its data sample points)
```shell
/data/Databases/LibriSpeech/Librispeech/train-clean-100
2518/154825/2518-154825-0025.flac       187280
2518/154825/2518-154825-0005.flac       236880
2518/154825/2518-154825-0056.flac       252640
```
Example: train_clean_100.ltr (each line is the character level transcript with | as blank)
```
W I L L | K E E P | T H E M | I N | C H E C K | A | L I T T L E | W H I L E | T H E Y | T R Y | T O | B R I N G | T H E | L I F E | B A C K | T O | S O | G O O D | A | M A N | B U T | I F | W E | F A L L | W H Y | W E | S H A L L | F A L L | T O G E T H E R | A N D | O U T W I T | T H E I R | C U N N I N G |
E V E R | S O | L I T T L E | I T | C A U S E D | M E | S H A R P | P A I N | F O R | F E E L I N G | W A S | C O M I N G | B A C K | A F T E R | T H E | F I R S T | N U M B N E S S | O F | T H E | S H O T | T H E Y | H A V E | B R O K E | T H E | L E G | T H O U G H | I T | B L E E D S | L I T T L E | E L Z E V I R | S A I D | W E | H A V E | N O | T I M E | T O | S P L I C E | I T | H E R E |
T H I N K I N G | T H A T | W E | W E R E | H I D I N G | B Y | T H E | S E A | F I V E | M I N U T E S | L A T E R | E L Z E V I R | S T E P P E D | O N | T O | T H E | C L I F F | T O P | W I T H | M E | U P O N | H I S | B A C K | W E | H A V E | M A D E | S O M E T H I N G | O F | T H I S | T H R O W | H E | S A I D | A N D | A R E | S A F E | F O R | A N O T H E R | H O U R | T H O U G H | I | T H O U G H T | T H Y | G I D D Y | H E A D | H A D | R U I N E D | U S |
```
dict.ltr.txt is the vocabulary.

When you have the kaldi style data for myst and ogi, you can try the script to convert the data to fairseq style:
```shell
run_prepare_myst.sh
run_prepare_ogi.sh
```

### 2. Added hyperparameters based on wav2vec2.0 code
The varaibles in the config file, e.g. ```exp/pretraining/myst_sp.yaml``` are used to control which part of the model is updated. You can look into the following files and search these variables for detailed implementations:
```shell
fairseq/models/wav2vec/wav2vec2.py
fairseq/models/wav2vec/wav2vec2_asr.py
```
The varaibles added are:
- bottleneck_dim: the dimension after down projection in residual adapter
- freeze_adapter: whether freeze adapter in adapatation or finetuning
- freeze_bacone: whether freeze backbone or not
- no_pretrained_weights and pretrained_weights_path: used for continual learning from a pretrained wav2vec2.0 model.
- use_first_adapter: whether adding the residual adapter after conv encoder 
- adapter_before_quant: adding adapter before the quantization layer in wav2vec2.0.

### 3. Domain responsible adaptation and finetuning
Given previous added variables, we can do domain responsible adaptation with
```shell
wav2vec_base_model=wav2vec_base/wav2vec_small.pt
this_dir=`pwd`
data_dir=$this_dir/myst_data/
exp_dir=exp/adaptation/

[ ! -d $exp_dir ] && mkdir -p $exp_dir
train_log=train.log

CUDA_VISIBLE_DEVICES="0,1" fairseq-hydra-train \
      --config-dir $this_dir/config/pretraining/ \
      --config-name myst_sp.yaml \
      task.data=${data_dir} \
      hydra.run.dir=${exp_dir} \
      common.log_file=${train_log} \
      model.bottleneck_dim=1024 \
      model.freeze_adapter=false \
      model.freeze_backbone=true \
      model.no_pretrained_weights=false \
      model.pretrained_weights_path=$this_dir/$wav2vec_base_model \
      model.use_first_adapter=true \
      model.adapter_before_quant=true #> $exp_dir/log 2>&1
```
where only the adapter is updated.

and domain responsible finetuning with:
```shell
wav2vec_base_model="the model obtained after adaptation"
CUDA_VISIBLE_DEVICES="0" fairseq-hydra-train \
    --config-dir ${this_dir}/config/finetuning \
    --config-name myst_sp.yaml \
    task.data=$data_dir \
    model.w2v_path=$this_dir/${wav2vec_base_model} \
    hydra.run.dir=${exp_dir} \
    common.log_file=${train_log} \
    model.bottleneck_dim=1024 \
    model.freeze_adapter=false \
    model.freeze_backbone=false \
    model.no_pretrained_weights=false \
    model.use_first_adapter=true \
    model.adapter_before_quant=true > $exp_dir/log 2>&1 &
```
where both the adapter and backbone encoder are updated.

You may check the example in ```run_myst.sh``` with 
- Stage 1: domain responsible adaptation
- Stage 2: domain responsible finetuning
- Stage 3: Viterbi-decoding

The yaml files are not exactly the same used in our paper, please try out new settings for own dataset. We suggest using similar learning rate to the pretraining stage in adaptation because residual adapters are randomly initialized. In the finetuning stage, use a smaller learning rate (1/10 of the adaptation lr could be a good start)

## Citations
```
@inproceedings{fan22d_interspeech,
  author={Ruchao Fan and Abeer Alwan},
  title={{DRAFT: A Novel Framework to Reduce Domain Shifting in Self-supervised Learning and Its Application to Children’s ASR}},
  year=2022,
  booktitle={Proc. Interspeech 2022},
  pages={4900--4904},
  doi={10.21437/Interspeech.2022-11128}
}
```

```
@article{fan2022towards,
  title={Towards better domain adaptation for self-supervised models: A case study of child asr},
  author={Fan, Ruchao and Zhu, Yunzheng and Wang, Jinhan and Alwan, Abeer},
  journal={IEEE Journal of Selected Topics in Signal Processing},
  volume={16},
  number={6},
  pages={1242--1252},
  year={2022},
  publisher={IEEE}
}
```

