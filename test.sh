python test.py --ckpt checkpoints/qm9_500steps/2000000.pt --config checkpoints/qm9_500steps/qm9_500steps.yml --test_set './data/GEOM/QM9/test_data_1k.pkl' --start_idx 800 --end_idx 1000 --sampling_type same_mask_noisy --n_steps 500 --device cuda:1 --w_global 0.1 --clip 1000 --clip_local 20 --global_start_sigma 5 --tag SubGDiff500

python test.py --ckpt checkpoints/qm9_200steps/2000000.pt --config checkpoints/qm9_200steps/qm9_200steps.yml --test_set './data/GEOM/QM9/test_data_1k.pkl' --start_idx 800 --end_idx 1000 --sampling_type same_mask_noisy --n_steps 200 --device cuda:0 --w_global 0.1 --clip 1000 --clip_local 20 --global_start_sigma 5 --tag SubGDiff200
 


