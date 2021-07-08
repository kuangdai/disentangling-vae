python main.py evae_mnist/kl_100ep_z10_e1.0_s0 -s 0 --checkpoint-every 25 -d mnist -e 100 -b 64 --lr 0.0005 -z 10 -l epsvae --epsvae-constrain-reconstruction False --epsvae-epsilon 10.0 --no-test
