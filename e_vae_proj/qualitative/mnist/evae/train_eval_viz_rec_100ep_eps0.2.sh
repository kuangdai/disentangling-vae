python main.py evae_mnist/rec_100ep_z10_e0.2_s0 -s 0 --checkpoint-every 25 -d mnist -e 100 -b 64 --lr 0.0005 -z 10 -l epsvae --epsvae-constrain-reconstruction --epsvae-epsilon 156.8 --no-test --record-loss-every=50 --pin-dataset-gpu 
python main.py evae_mnist/rec_100ep_z10_e0.2_s0 --is-eval-only 
python main_viz.py evae_mnist/rec_100ep_z10_e0.2_s0 all --is-show-loss --is-posterior -s 0 
python main_viz.py evae_mnist/rec_100ep_z10_e0.2_s0 traversals --is-show-loss -s 0 
