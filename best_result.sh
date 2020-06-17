python train.py --exp_name best   \
                --gpu_n 0         \
                --lr 0.001        \
                --optim adam      \
                --loss_func cross \
                --scheduler reduce \
                --model_name resnet34_wo_first_pool_dropout \
                --epoch 100 --weight_decay 0.0001  \
                --cutmix 1    


