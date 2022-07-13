
stage=2
end_stage=3

current_dir=`pwd`
#checkpoint_path=$current_dir/hubert_base/hubert_base_ls960.pt
exp_dir=$current_dir/mfcc_kmeans_libri/   #hubert_kmeans_libri/
feat_dir=$exp_dir/feat_layer6/
lab_dir=$exp_dir/lab_layer6/
km_path=$exp_dir/k_model.pt

if [ $stage -le 0 ] && [ $end_stage -ge 0 ]; then
    echo "Data Preparation for K-means"
    
    layer=6
    nshard=4
   
    for split in train valid; do
        for ((rank=0; rank<$nshard; rank++)); do
            echo "Extract MFCC and Feature"
            CUDA_VISIBLE_DEVICES=3 python simple_kmeans/dump_mfcc_feature.py $current_dir/libri_data $split $nshard $rank $feat_dir
            #echo "Extract Hubert Feature"
            #python simple_kmeans/dump_hubert_feature.py $current_dir/libri_data $split $checkpoint_path $layer $nshard $rank $feat_dir
        done
    done
fi

if [ $stage -le 1 ] && [ $end_stage -ge 1 ]; then
    echo "K-means Clustering"
    nshard=4
    n_cluster=100
    percent=-1 #use all data
    python simple_kmeans/learn_kmeans.py $feat_dir train $nshard $km_path $n_cluster --percent $percent --batch_size 10000
fi

if [ $stage -le 2 ] && [ $end_stage -ge 2 ]; then
    nshard=4

    for split in train valid; do
        for ((rank=0; rank<$nshard; rank++)); do
            echo "K-means Application"
            python simple_kmeans/dump_km_label.py $feat_dir $split $km_path $nshard $rank $lab_dir
        done

        echo "merge shards"
        for rank in $(seq 0 $((nshard-1))); do
            cat $lab_dir/${split}_${rank}_${nshard}.km
        done > $lab_dir/${split}.km
    done
fi

if [ $stage -le 3 ] && [ $end_stage -ge 3 ]; then
    n_clusters=100
    echo "create dummy dictionary"
    for x in $(seq 0 $((n_clusters - 1))); do
        echo "$x 1"
    done >> $lab_dir/dict.km.txt
fi

