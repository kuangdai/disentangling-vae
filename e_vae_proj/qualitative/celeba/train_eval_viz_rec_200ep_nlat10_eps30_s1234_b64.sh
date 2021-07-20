python main.py evae_celeba/rec_200ep_z10_e30_s1234_lr0.001_incr2 -s 1234 --checkpoint-every 25 -d celeba -e 200 -b 64 --lr 0.001 -z 10 -l epsvae --epsvae-constrain-reconstruction --epsvae-epsilon 30 --epsvae-interval-incr-L 2 --no-test --record-loss-every=50 --pin-dataset-gpu 
python main.py evae_celeba/rec_200ep_z10_e30_s1234 --is-eval-only 
python main_viz.py evae_celeba/rec_200ep_z10_e30_s1234 all --is-show-loss --is-posterior -s 1234 
python main_viz.py evae_celeba/rec_200ep_z10_e30_s1234 traversals --is-show-loss -s 1234 
python main.py evae_celeba/rec_200ep_z10_e30_s1234_lr0.0005_incr4 -s 1234 --checkpoint-every 25 -d celeba -e 200 -b 64 --lr 0.0005 -z 10 -l epsvae --epsvae-constrain-reconstruction --epsvae-epsilon 30 --epsvae-interval-incr-L 4 --no-test --record-loss-every=50 --pin-dataset-gpu 
python main.py evae_celeba/rec_200ep_z10_e30_s1234 --is-eval-only 
python main_viz.py evae_celeba/rec_200ep_z10_e30_s1234 all --is-show-loss --is-posterior -s 1234 
python main_viz.py evae_celeba/rec_200ep_z10_e30_s1234 traversals --is-show-loss -s 1234 
python main.py evae_celeba/rec_200ep_z10_e30_s1234_lr0.0005_incr2 -s 1234 --checkpoint-every 25 -d celeba -e 200 -b 64 --lr 0.0005 -z 10 -l epsvae --epsvae-constrain-reconstruction --epsvae-epsilon 30 --epsvae-interval-incr-L 2 --no-test --record-loss-every=50 --pin-dataset-gpu 
python main.py evae_celeba/rec_200ep_z10_e30_s1234 --is-eval-only 
python main_viz.py evae_celeba/rec_200ep_z10_e30_s1234 all --is-show-loss --is-posterior -s 1234 
python main_viz.py evae_celeba/rec_200ep_z10_e30_s1234 traversals --is-show-loss -s 1234 
python main.py evae_celeba/rec_200ep_z10_e30_s1234_lr0.001_incr4 -s 1234 --checkpoint-every 25 -d celeba -e 200 -b 64 --lr 0.001 -z 10 -l epsvae --epsvae-constrain-reconstruction --epsvae-epsilon 30 --epsvae-interval-incr-L 4 --no-test --record-loss-every=50 --pin-dataset-gpu 
python main.py evae_celeba/rec_200ep_z10_e30_s1234 --is-eval-only 
python main_viz.py evae_celeba/rec_200ep_z10_e30_s1234 all --is-show-loss --is-posterior -s 1234 
python main_viz.py evae_celeba/rec_200ep_z10_e30_s1234 traversals --is-show-loss -s 1234 
python main.py evae_celeba/rec_200ep_z10_e30_s1234_lr0.001_incr2 -s 1234 --checkpoint-every 25 -d celeba -e 200 -b 64 --lr 0.001 -z 10 -l epsvae --epsvae-constrain-reconstruction --epsvae-epsilon 30 --epsvae-interval-incr-L 2 --no-test --record-loss-every=50 --pin-dataset-gpu 
python main.py evae_celeba/rec_200ep_z10_e30_s1234 --is-eval-only 
python main_viz.py evae_celeba/rec_200ep_z10_e30_s1234 all --is-show-loss --is-posterior -s 1234 
python main_viz.py evae_celeba/rec_200ep_z10_e30_s1234 traversals --is-show-loss -s 1234 
