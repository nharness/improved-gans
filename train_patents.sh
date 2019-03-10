export word=patents
export pixels=512
echo $HOSTNAME >> ${word}.record.txt
python train_${word}.py --dataset imagenet_train --is_train True --checkpoint_dir gan/checkpoint_${word} --image_size ${pixels} --is_crop True --sample_dir gan/samples_${word} --image_width ${pixels} --batch_size 16
