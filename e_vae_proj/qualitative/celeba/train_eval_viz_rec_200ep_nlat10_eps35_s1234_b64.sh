python main.py evae_celeba/rec_200ep_z10_e35_s1234_lr5e-05_incr4 -s 1234 --checkpoint-every 25 -d celeba -e 200 -b 64 --lr 5e-05 -z 10 -l epsvae --epsvae-constrain-reconstruction --epsvae-epsilon 35 --epsvae-interval-incr-L 4 --no-test --record-loss-every=50 --pin-dataset-gpu 
python main.py evae_celeba/rec_200ep_z10_e35_s1234_lr5e-05_incr4 --is-eval-only 
python main_viz.py evae_celeba/rec_200ep_z10_e35_s1234_lr5e-05_incr4 all --is-show-loss --is-posterior -s 1234 
python main_viz.py evae_celeba/rec_200ep_z10_e35_s1234_lr5e-05_incr4 traversals --is-show-loss -s 1234 
python main.py evae_celeba/rec_200ep_z10_e35_s1234_lr0.0001_incr4 -s 1234 --checkpoint-every 25 -d celeba -e 200 -b 64 --lr 0.0001 -z 10 -l epsvae --epsvae-constrain-reconstruction --epsvae-epsilon 35 --epsvae-interval-incr-L 4 --no-test --record-loss-every=50 --pin-dataset-gpu 
python main.py evae_celeba/rec_200ep_z10_e35_s1234_lr0.0001_incr4 --is-eval-only 
python main_viz.py evae_celeba/rec_200ep_z10_e35_s1234_lr0.0001_incr4 all --is-show-loss --is-posterior -s 1234 
python main_viz.py evae_celeba/rec_200ep_z10_e35_s1234_lr0.0001_incr4 traversals --is-show-loss -s 1234 
python main.py evae_celeba/rec_200ep_z10_e35_s1234_lr0.0003_incr4 -s 1234 --checkpoint-every 25 -d celeba -e 200 -b 64 --lr 0.0003 -z 10 -l epsvae --epsvae-constrain-reconstruction --epsvae-epsilon 35 --epsvae-interval-incr-L 4 --no-test --record-loss-every=50 --pin-dataset-gpu 
python main.py evae_celeba/rec_200ep_z10_e35_s1234_lr0.0003_incr4 --is-eval-only 
python main_viz.py evae_celeba/rec_200ep_z10_e35_s1234_lr0.0003_incr4 all --is-show-loss --is-posterior -s 1234 
python main_viz.py evae_celeba/rec_200ep_z10_e35_s1234_lr0.0003_incr4 traversals --is-show-loss -s 1234 
