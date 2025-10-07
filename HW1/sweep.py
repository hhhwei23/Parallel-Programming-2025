#!/usr/bin/env python3
import argparse, os, subprocess, sys, csv, shlex, time
from pathlib import Path

def get_file_size(path):
    return Path(path).stat().st_size

def run_one(nodes, procs, exe, N, inp, outpath, extra_srun):
    cmd = ["srun", "-N", str(nodes), "-n", str(procs)]
    if extra_srun:
        cmd += shlex.split(extra_srun)
    cmd += [exe, str(N), inp, outpath]
    print("[RUN]", " ".join(shlex.quote(x) for x in cmd))
    cp = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if cp.returncode != 0:
        sys.stderr.write(cp.stderr)
        raise RuntimeError(f"srun failed (rc={cp.returncode})")
    # 從 stdout 擷取 'hw1,...' 行
    line = None
    for ln in cp.stdout.splitlines():
        if ln.startswith("hw1,"):
            line = ln.strip()
    if line is None:
        sys.stderr.write(cp.stdout)
        raise RuntimeError("No 'hw1,...' line found in stdout")
    # 覆寫 nodes 欄位（第4欄）
    parts = line.split(",")
    if len(parts) < 9:
        raise RuntimeError(f"Malformed CSV line: {line}")
    parts[3] = str(nodes)
    return ",".join(parts)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default="./hw1")
    ap.add_argument("--input", required=True, help="input .bin (float32)")
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--csv", default="results.csv")
    ap.add_argument("--nodes", nargs="+", type=int, required=True)
    ap.add_argument("--procs", nargs="+", type=int, required=True)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--srun-extra", default="", help="extra flags, e.g. '--partition=cuda --time=00:10:00'")
    ap.add_argument("--N", type=int, default=None, help="override N (floats); default: infer from file size")
    args = ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    # 設定 CSV header（若檔案不存在）
    new_csv = not Path(args.csv).exists()
    if new_csv:
        with open(args.csv, "w", newline="") as f:
            f.write("case,N,P,nodes,io_avg,comm_avg,comp_avg,total_wall,iterations\n")

    # 計算 N（若未指定）
    if args.N is None:
        bytes_ = get_file_size(args.input)
        if bytes_ % 4 != 0:
            print(f"[ERROR] input size {bytes_} not multiple of 4 (float32)", file=sys.stderr)
            sys.exit(2)
        N = bytes_ // 4
    else:
        N = args.N
    print(f"[INFO] Using N={N} from {'--N' if args.N else 'file size'}; input={args.input}")

    # 跑 sweep
    with open(args.csv, "a", newline="") as f:
        for nodes in args.nodes:
            for procs in args.procs:
                for r in range(1, args.repeat+1):
                    outpath = str(Path(args.outdir)/f"out_N{nodes}_P{procs}_r{r}.bin")
                    try:
                        row = run_one(nodes, procs, args.exe, N, args.input, outpath, args.srun_extra)
                        f.write(row + "\n")
                        f.flush()
                    except Exception as e:
                        print(f"[WARN] nodes={nodes} P={procs} r={r}: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()


"""
testcase file paths
/home/pp25/share/hw1/testcases

python3 sweep.py --exe ./hw1 --input testcase/30.in --nodes 1 --procs 1 2 3 4 5 6 7 8 9 10 11 12 --repeat 1 --outdir testcase/30.out --csv results_N1_n12_t30_peocessor.csv

python3 sweep.py --exe ./hw1 --input testcase/30.in --nodes 1 2 3 4 --procs 12 --repeat 1 --outdir testcase/30.out --csv results_N1_n12_t30_nodes.csv
"""