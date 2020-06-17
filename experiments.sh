python train.py --gpu_n 0 --lr 0.001 --optim adam --loss_func cross --scheduler reduce --model_name resnet18 --exp_name resnet18_vanil

python train.py --gpu_n 0 --lr 0.001 --optim adam --loss_func cross --scheduler reduce --model_name resnet34 --exp_name resnet34_vanil

python train.py --gpu_n 0 --lr 0.001 --optim adam --loss_func cross --scheduler reduce --model_name resnet18_wo_first_pool --exp_name resnet18_wo_first_pool

python train.py --gpu_n 0 --lr 0.001 --optim adam --loss_func cross --scheduler reduce --model_name resnet18_wo_first_pool_dropout --exp_name resnet18_wo_first_pool_dropout

python train.py --gpu_n 1 --lr 0.001 --optim adam --loss_func smooth --scheduler reduce --model_name resnet18 --exp_name resnet18_vanil_smooth

python train.py --gpu_n 0 --lr 0.001 --optim adam --loss_func cross --scheduler reduce --model_name resnet34_wo_first_pool_dropout --exp_name cutmix \
                --epoch 100 --weight_decay 0.0001 --cutmix

python train.py --gpu_n 0 --lr 0.001 --optim adam --loss_func cross --scheduler reduce --model_name resnet18_wo_first_pool_dropout --exp_name cutmix_mixup \
                --epoch 100 --weight_decay 0.0001 --cutmix

python train.py --gpu_n 0 --lr 0.001 --optim radam --loss_func cross --scheduler reduce --model_name resnet18_wo_first_pool_dropout --exp_name cutmix_radam \
                --epoch 100 --weight_decay 0.0001

python train.py --gpu_n 0 --lr 0.1 --optim sgd --loss_func cross --scheduler reduce --model_name resnet18_wo_first_pool_dropout --exp_name cutmix_sgd \
                --epoch 100 --weight_decay 0.0001 --patience 8 --lr_factor 0.5 --cutmix

python train.py --gpu_n 1 --lr 0.1 --optim sgd --loss_func cross --scheduler reduce --model_name resnet18_wo_first_pool_dropout --exp_name cutmix_sgd_autoaug \
                --epoch 150 --weight_decay 0.0001 --auto_aug 1 --cutmix