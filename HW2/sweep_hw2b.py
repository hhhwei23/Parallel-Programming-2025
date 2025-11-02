#!/usr/bin/env python3
import os, subprocess, sys, csv, shlex, glob, shutil
from pathlib import Path

# ===== 測資 =====
ITERS  = 10000
LEFT   = -0.2931209325179713
RIGHT  = -0.2741427339562606
LOWER  = -0.6337125743279389
UPPER  = -0.6429654881215695
WIDTH  = 7680
HEIGHT = 4320

EXE            = "./hw2b"
NODES          = 2
NPROCS_LIST    = [1, 2, 3, 4]   # ← 掃不同 MPI processes
THREADS_LIST   = [6]
REPEAT         = 1

SRUN_EXTRA     = "--cpu-bind=cores"
OUTDIR         = "outputs_hw2b"
TIMES_CSV      = "times_scan_hw2b.csv"

def run_one(procs, threads, rep):
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)
    env["OMP_PROC_BIND"]   = "true"
    env["OMP_PLACES"]      = "cores"

    out_png = f"out_p{procs}_t{threads}_r{rep}.png"

    cmd = ["srun", "-N", str(NODES), "-n", str(procs), "-c", str(threads)]
    if SRUN_EXTRA:
        cmd += shlex.split(SRUN_EXTRA)
    cmd += [EXE, out_png, str(ITERS),
            str(LEFT), str(RIGHT), str(LOWER), str(UPPER),
            str(WIDTH), str(HEIGHT)]

    print("[RUN]", " ".join(shlex.quote(x) for x in cmd))
    cp = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    if cp.returncode != 0:
        sys.stderr.write(cp.stderr)
        raise RuntimeError(f"srun failed (rc={cp.returncode})")

    # 讀 prof_summary.csv（第 2 行資料），並搬運紀錄檔
    ps = Path("prof_summary.csv")
    if not ps.exists():
        raise RuntimeError("prof_summary.csv not found (did hw2b write it on rank 0?)")
    lines  = [ln.strip() for ln in ps.read_text().splitlines() if ln.strip()]
    header = lines[0].split(",")
    data   = lines[1].split(",")

    # 收檔：每次 run 建一個資料夾
    profdir = Path("profiles") / f"p{procs}_t{threads}_r{rep}"
    profdir.mkdir(parents=True, exist_ok=True)

    for f in glob.glob("prof_rank*.csv"):   # 每個 thread 的 loading
        shutil.move(f, profdir / Path(f).name)

    pr = Path("prof_ranks.csv")             # 每個 rank 的彙整（如果你有加輸出）
    if pr.exists():
        shutil.move(pr, profdir / "prof_ranks.csv")

    shutil.copyfile(ps, profdir / "prof_summary.csv")

    Path(OUTDIR).mkdir(parents=True, exist_ok=True)
    op = Path(out_png)
    if op.exists():
        shutil.move(out_png, str(Path(OUTDIR) / out_png))

    return header, data

def main():
    header_written = Path(TIMES_CSV).exists()
    with open(TIMES_CSV, "a", newline="") as fcsv:
        writer = csv.writer(fcsv)
        for procs in NPROCS_LIST:
            for threads in THREADS_LIST:
                for r in range(1, REPEAT+1):
                    hdr, row = run_one(procs, threads, r)
                    if not header_written:
                        writer.writerow(["repeat"] + hdr)
                        header_written = True
                    writer.writerow([str(r)] + row)

    print(f"[OK] 已寫入 {TIMES_CSV}，詳細檔案在 profiles/p*_t*_r*/ 與 {OUTDIR}/")

if __name__ == "__main__":
    main()
