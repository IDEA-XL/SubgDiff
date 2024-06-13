python train_dist.py --config ./configs/qm9_200steps.yml --device cuda:0 --logdir ./logs --tag qm9_500steps --n_jobs 12 --print_freq 200
python train_dist.py --config ./configs/qm9_500steps.yml --device cuda:1 --logdir ./logs --tag qm9_200steps --n_jobs 12 --print_freq 200
