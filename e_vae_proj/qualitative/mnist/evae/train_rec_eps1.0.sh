python main.py evae_mnist/rec_100ep_z10_e1.0_s0 -s 0 --checkpoint-every 25 -d mnist -e 100 -b 64 --lr 0.0005 -z 10 -l epsvae --epsvae-constrain-reconstruction True --epsvae-epsilon 4096.0 --no-test
