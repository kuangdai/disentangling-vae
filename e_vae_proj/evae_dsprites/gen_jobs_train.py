import numpy as np
from pathlib import Path
import sys

if __name__ == '__main__':
    # constraint type
    cons = sys.argv[1]
    assert cons in ['kl', 'rec']

    # absolute path
    my_path = Path(__file__).parent.resolve().expanduser()
    main_path = my_path.parent.parent

    # read
    epses = np.loadtxt(my_path / f'grid_eps_{cons}')
    nlats = np.loadtxt(my_path / 'grid_nlats')
    seed = 0

    # cmd template
    cmd_tmp = f'singularity exec --nv {str(main_path)}/../torch.simg ' \
              f'python main.py evae_dsprites_{cons}/z%d_e%s_s{seed} -s {seed} ' \
              f'--checkpoint-every 25 -d dsprites -e 50 -b 256 --lr 0.0005 ' \
              f'-z %d -l epsvae --epsvae-constrain-reconstruction {cons == "rec"} ' \
              f'--epsvae-epsilon %s --is-metrics --no-test --no-progress-bar\n'

    # job header
    with open(my_path / 'job_header', 'r') as f:
        job_header = f.read()

    # jobs grouped by eps
    for ieps, eps in enumerate(epses):
        with open(my_path / f'jobs_{cons}/eps{ieps}.job', 'w') as f:
            h = job_header.replace('@JNAME@', f'{cons}_e{ieps}')
            h = h.replace('@OUTPUT@', str(my_path / f'jobs_{cons}/eps{ieps}.slurm'))
            h = h.replace('@MAINPATH@', str(main_path))
            f.write(h)
            for nlat in nlats:
                if cons == 'rec':
                    unnormalized_eps = eps * 64 * 64
                else:
                    unnormalized_eps = eps * nlat
                cmd = cmd_tmp % (nlat, str(eps), nlat, str(unnormalized_eps))
                f.write(cmd)

    # submit
    with open(my_path / f'jobs_{cons}/submit0.sh', 'w') as f:
        for ieps, eps in enumerate(epses):
            if ieps % 2 == 0:
                f.write(f'sbatch {str(my_path)}/jobs_{cons}/eps{ieps}.job\n')

    with open(my_path / f'jobs_{cons}/submit1.sh', 'w') as f:
        for ieps, eps in enumerate(epses):
            if ieps % 2 == 1:
                f.write(f'sbatch {str(my_path)}/jobs_{cons}/eps{ieps}.job\n')
