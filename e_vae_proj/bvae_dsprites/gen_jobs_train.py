import numpy as np
from pathlib import Path

if __name__ == '__main__':
    # absolute path
    my_path = Path(__file__).parent.resolve().expanduser()
    main_path = my_path.parent.parent

    # read
    betas = np.loadtxt(my_path / 'grid_betas')
    nlats = np.loadtxt(my_path / 'grid_nlats')
    seed = 0

    # cmd template
    cmd_tmp = f'singularity exec {str(main_path)}/../torch.simg ' \
              f'python main.py bvae_dsprites/z%d_b%s_s{seed} -s {seed} ' \
              f'--checkpoint-every 25 -d dsprites -e 50 -b 256 --lr 0.01 ' \
              f'-z %d -l betaH --betaH-B %s --is-metrics --no-test --no-progress-bar\n'

    # job header
    with open(my_path / 'job_header', 'r') as f:
        job_header = f.read()

    # jobs grouped by beta
    for ibeta, beta in enumerate(betas):
        with open(my_path / f'jobs/beta{ibeta}.job', 'w') as f:
            h = job_header.replace('@JNAME@', f'bvae_dsprites_beta{ibeta}')
            h = h.replace('@OUTPUT@', str(my_path / f'jobs/beta{ibeta}.slurm'))
            h = h.replace('@MAINPATH@', str(main_path))
            f.write(h)
            for nlat in nlats:
                unnormalized_beta = beta * 64 * 64 / nlat
                cmd = cmd_tmp % (nlat, str(beta), nlat, str(unnormalized_beta))
                f.write(cmd)

    # submit
    with open(my_path / 'jobs/submit0.sh', 'w') as f:
        for ibeta, beta in enumerate(betas):
            if ibeta % 2 == 0:
                f.write(f'sbatch {str(my_path)}/jobs/beta{ibeta}.job\n')

    with open(my_path / 'jobs/submit1.sh', 'w') as f:
        for ibeta, beta in enumerate(betas):
            if ibeta % 2 == 1:
                f.write(f'sbatch {str(my_path)}/jobs/beta{ibeta}.job\n')
