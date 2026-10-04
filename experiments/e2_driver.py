import subprocess, sys, json
part=sys.argv[1]; timeout=int(sys.argv[2]); grid=json.loads(sys.argv[3])
for fam,n,seed in grid:
    try:
        r=subprocess.run(['python3','experiments/e2_worker.py',part,fam,str(n),str(seed)],timeout=timeout,capture_output=True,text=True)
        print(fam,n,seed,'ok' if r.returncode==0 else r.stderr[-300:],flush=True)
    except subprocess.TimeoutExpired:
        with open(f'results/e2{part}_timeouts.jsonl','a') as fh: fh.write(json.dumps(dict(family=fam,n=n,seed=seed,timeout_s=timeout))+'\n')
        print(fam,n,seed,'TIMEOUT',flush=True)
