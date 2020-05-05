config = {
    "node": 8,
    "ntasks-per-node": 8,
    "cpus-per-task": 1,
    "time": "5:00:00",
    "mem": "4GB",
    "job-name": "",
    "mail-type": "END",
    "mail-user": "cy1505@nyu.edu",
    "output": ""
}

sheba = "#!/bin/bash"
moduleinfo = "module purge\nmodule load openmpi/gnu/4.0.2\n"

## strong
N = 6400
cnt = 5
idx = [2**i for i in range(0, cnt)]
pArr = [i**2 for i in idx]
lnArr = [int(N/i) for i in idx]
maxiter = 20000

for (p, ln) in zip(pArr, lnArr):
    fname = "jacobi-strong-ln%s-N%s-p%s" % (ln, N, p)
    config['job-name'] = fname
    config['output'] = fname+'.out'
    with open("./%s.sh" %fname, 'w') as f:
        f.write('\n'.join(["#SBATCH --%s=%s" %(k,v) for k,v in config.items()]))
        f.write('\n\n')
        f.write(moduleinfo)
        f.write("mpiexec %s %s %s" %("jacobi-mpi", ln, maxiter))

## weak
cnt = 5
ln = 100
ratio = 50
idx = [2**i for i in range(0, cnt)]
pArr = [i**2 for i in idx]
nArr = [i*ratio for i in idx]
maxiter = 20000

for (p,N) in zip(pArr, nArr):
    fname = "jacobi-weak-ln%s-N%s-p%s" % (ln, N, p)
    config['job-name'] = fname
    config['output'] = fname+'.out'
    with open("./%s.sh" %fname, 'w') as f:
        f.write('\n'.join(["#SBATCH --%s=%s" %(k,v) for k,v in config.items()]))
        f.write('\n\n')
        f.write(moduleinfo)
        f.write("mpiexec %s %s %s" %("jacobi-mpi", ln, maxiter))

## ssort

sn = [int(1e4), int(1e5), int(1e6)]

for num in sn:
    fname = 'ssort-1e'+str(str(num).count('0'))
    config['job-name'] = fname
    config['output'] = fname+'.out'
    with open("./%s.sh" %fname, 'w') as f:
        f.write('\n'.join(["#SBATCH --%s=%s" %(k,v) for k,v in config.items()]))
        f.write('\n\n')
        f.write(moduleinfo)
        f.write("mpiexec %s %s" %("ssort", num))
