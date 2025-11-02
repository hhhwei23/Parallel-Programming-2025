#!/usr/bin/env python3
import os, subprocess, sys, csv, shlex, glob, shutil
from pathlib import Path

# ===== 你的測資 =====
ITERS  = 10000
LEFT   = -0.2931209325179713
RIGHT  = -0.2741427339562606
LOWER  = -0.6337125743279389
UPPER  = -0.6429654881215695
WIDTH  = 7680
HEIGHT = 4320

# ===== 這裡改成 pthread 版本 =====
EXE            = "./hw2a"          # 你的 pthread 可執行檔
NODES          = 1                 # 只跑單一 process（pthread 內多執行緒）
TASKS          = 1                 # 等同 -n 1
THREADS_LIST   = [1, 2, 4, 8, 12]  # 要掃的執行緒數
REPEAT         = 1

# 建議綁核心，比較穩定（Slurm）
SRUN_EXTRA     = "--cpu-bind=cores"
OUTDIR         = "outputs_hw2a"
TIMES_CSV      = "times_scan_hw2a.csv"

def run_one(threads, rep):
    env = os.environ.copy()
    # 你的程式用 NTHREADS 控制 pthread 執行緒數
    env["NTHREADS"] = str(threads)

    out_png = f"out_t{threads}_r{rep}.png"

    # 在 Slurm 下用 -c threads 預留同數量 CPU；-n 維持 1
    cmd = ["srun", "-N", str(NODES), "-n", str(TASKS), "-c", str(threads)]
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

    # 讀取你的程式輸出的彙總檔（含 compute_s / total_s / io_s）
    ps = Path("prof_summary.csv")
    if not ps.exists():
        raise RuntimeError("prof_summary.csv not found (did hw2a write it?)")
    lines  = [ln.strip() for ln in ps.read_text().splitlines() if ln.strip()]
    header = lines[0].split(",")
    data   = lines[1].split(",")

    # 收 thread loading（每個 thread 的時間與工作量）
    pt = Path("prof_threads.csv")
    if not pt.exists():
        raise RuntimeError("prof_threads.csv not found (did hw2a write it?)")

    # 每次 run 建一個資料夾保存
    profdir = Path("profiles_hw2a") / f"t{threads}_r{rep}"
    profdir.mkdir(parents=True, exist_ok=True)
    shutil.move(str(ps), profdir / "prof_summary.csv")
    shutil.move(str(pt), profdir / "prof_threads.csv")

    # 也把 png 移走
    Path(OUTDIR).mkdir(parents=True, exist_ok=True)
    op = Path(out_png)
    if op.exists():
        shutil.move(out_png, str(Path(OUTDIR) / out_png))

    return header, data  # 回傳 prof_summary 的欄位與資料列

def main():
    new_csv = not Path(TIMES_CSV).exists()
    with open(TIMES_CSV, "a", newline="") as fcsv:
        writer = csv.writer(fcsv)
        for t in THREADS_LIST:
            for r in range(1, REPEAT+1):
                hdr, row = run_one(t, r)
                # 首次把 header 寫進來，多加一欄 threads / repeat 便於後續繪圖
                if new_csv:
                    writer.writerow(["threads", "repeat"] + hdr)
                    new_csv = False
                writer.writerow([str(t), str(r)] + row)

    print(f"[OK] 已寫入 {TIMES_CSV}；每次 run 的檔案在 profiles_hw2a/t*_r*/，圖片在 {OUTDIR}/")

if __name__ == "__main__":
    main()
