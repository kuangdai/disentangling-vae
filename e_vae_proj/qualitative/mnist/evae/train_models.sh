# check epoch 100 vs 400
python main.py evae_mnist/kl_100ep_z10_e0.1_s0 -s 0 --checkpoint-every 25 -d mnist -e 100 -b 64 --lr 0.0005 -z 10 -l epsvae --epsvae-constrain-kl-divergence --epsvae-epsilon 1.0 --no-test --record-loss-every=50 --pin-dataset-gpu &
python main.py evae_mnist/kl_400ep_z10_e0.1_s0 -s 0 --checkpoint-every 25 -d mnist -e 400 -b 64 --lr 0.0005 -z 10 -l epsvae --epsvae-constrain-kl-divergence --epsvae-epsilon 1.0 --no-test --record-loss-every=50 --pin-dataset-gpu &
