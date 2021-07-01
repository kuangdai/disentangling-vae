import numpy as np
from pathlib import Path

if __name__ == '__main__':
    # absolute path
    my_path = Path(__file__).parent.resolve().expanduser()
    main_path = my_path.parent.parent

    # read
    epsilons = np.loadtxt(my_path / 'grid_epsilons')
    nlats = np.loadtxt(my_path / 'grid_nlats')
    seed = 0

    # cmd template
    cmd_tmp = f'singularity exec --nv {str(main_path)}/../torch.simg ' \
              f'python main.py evae_dsprites/z%d_b%s_s{seed} -s {seed} ' \
              f'--checkpoint-every 25 -d dsprites -e 50 -b 256 --lr 0.0005 ' \
              f'-z %d -l epsilonH --epsilonH-B %s --is-metrics --no-test --no-progress-bar\n'

    # job header
    with open(my_path / 'job_header', 'r') as f:
        job_header = f.read()

    # jobs grouped by epsilon
    for iepsilon, epsilon in enumerate(epsilons):
        with open(my_path / f'jobs/epsilon{iepsilon}.job', 'w') as f:
            h = job_header.replace('@JNAME@', f'bvae_dsprites_epsilon{iepsilon}')
            h = h.replace('@OUTPUT@', str(my_path / f'jobs/epsilon{iepsilon}.slurm'))
            h = h.replace('@MAINPATH@', str(main_path))
            f.write(h)
            for nlat in nlats:
                unnormalized_epsilon = epsilon * 64 * 64 / nlat
                cmd = cmd_tmp % (nlat, str(epsilon), nlat, str(unnormalized_epsilon))
                f.write(cmd)

    # submit
    with open(my_path / 'jobs/submit0.sh', 'w') as f:
        for iepsilon, epsilon in enumerate(epsilons):
            if iepsilon % 2 == 0:
                f.write(f'sbatch {str(my_path)}/jobs/epsilon{iepsilon}.job\n')

    with open(my_path / 'jobs/submit1.sh', 'w') as f:
        for iepsilon, epsilon in enumerate(epsilons):
            if iepsilon % 2 == 1:
                f.write(f'sbatch {str(my_path)}/jobs/epsilon{iepsilon}.job\n')
