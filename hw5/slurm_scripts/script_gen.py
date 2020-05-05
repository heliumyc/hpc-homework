config = {
    "nodes": 8,
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

import math

## strong
N = 6400
cnt = 5
idx = [2**i for i in range(0, cnt)]
pArr = [i**2 for i in idx]
lnArr = [int(N/i) for i in idx]
maxiter = 20000

strongfiles = []
for (p, ln) in zip(pArr, lnArr):
    fname = "jacobi-strong-ln%s-N%s-p%s" % (ln, N, p)
    strongfiles.append(fname+'.sh')
    config['job-name'] = fname
    config['output'] = fname+'.out'
    config['nodes'] = int(math.sqrt(p))
    config['ntasks-per-node'] = int(math.sqrt(p))
    with open("./%s.sh" %fname, 'w') as f:
        f.write(sheba+'\n')
        f.write('\n'.join(["#SBATCH --%s=%s" %(k,v) for k,v in config.items()]))
        f.write('\n\n')
        f.write(moduleinfo)
        f.write("mpiexec %s %s %s" %("../jacobi-mpi", ln, maxiter))

## weak
cnt = 5
ln = 100
idx = [2**i for i in range(0, cnt)]
pArr = [i**2 for i in idx]
maxiter = 20000

weakfiles = []
for p in pArr:
    N = int(math.sqrt(p))*ln
    fname = "jacobi-weak-ln%s-N%s-p%s" % (ln, N, p)
    weakfiles.append(fname+'.sh')
    config['job-name'] = fname
    config['output'] = fname+'.out'
    config['nodes'] = int(math.sqrt(p))
    config['ntasks-per-node'] = int(math.sqrt(p))
    with open("./%s.sh" %fname, 'w') as f:
        f.write(sheba+'\n')
        f.write('\n'.join(["#SBATCH --%s=%s" %(k,v) for k,v in config.items()]))
        f.write('\n\n')
        f.write(moduleinfo)
        f.write("mpiexec %s %s %s" %("../jacobi-mpi", ln, maxiter))

## ssort

sn = [int(1e4), int(1e5), int(1e6)]

sortfiles = []
for num in sn:
    fname = 'ssort-1e'+str(str(num).count('0'))
    sortfiles.append(fname+'.sh')
    config['job-name'] = fname
    config['output'] = fname+'.out'
    config['nodes'] = int(math.sqrt(p))
    config['ntasks-per-node'] = int(math.sqrt(p))
    with open("./%s.sh" %fname, 'w') as f:
        f.write(sheba+'\n')
        f.write('\n'.join(["#SBATCH --%s=%s" %(k,v) for k,v in config.items()]))
        f.write('\n\n')
        f.write(moduleinfo)
        f.write("mpiexec %s %s" %("../ssort", num))

with open('./runsort.sh', 'w') as f:
    f.write(sheba+'\n')
    for name in sortfiles:
        f.write('sbatch '+name+'\n')
with open('./runweak.sh', 'w') as f:
    f.write(sheba+'\n')
    for name in weakfiles:
        f.write('sbatch '+name+'\n')
with open('./runstrong.sh', 'w') as f:
    f.write(sheba+'\n')
    for name in strongfiles:
        f.write('sbatch '+name+'\n')