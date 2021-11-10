python main.py \
    --mode train \
    --gpu 1 \
    --model PWCLO_Net \
    --data_root /tmp/data_odometry_velodyne/dataset \
    --log_dir pwclo_net_log_ \
    --result_dir result \
    --train_list 0 1 2 3 4 5 6 \
    --val_list 7 8 9 10 \
    --test_list 0 1 2 3 4 5 6 7 8 9 10 \
    --num_point 8192 \
    --max_epoch 300 \
    --learning_rate 0.001 \
    --batch_size 8 \
    > log_train_PWCLO_net.txt 2>&1 &
