import argparse, subprocess, sys, os, json, time

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["mag","cossin","complex"], default="complex")
    ap.add_argument("--N", type=int, default=128)
    ap.add_argument("--train_sizes", type=str, default="500,1000,2000,5000,10000,20000")
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--run_dir", type=str, default="runs")
    args = ap.parse_args()

    train_sizes = [int(x) for x in args.train_sizes.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    results = []
    for ts in train_sizes:
        for seed in seeds:
            cmd = [
                sys.executable, "-m", "src.train",
                "--model", args.model,
                "--N", str(args.N),
                "--train_size", str(ts),
                "--val_size", "2000",
                "--test_size", "5000",
                "--epochs", str(args.epochs),
                "--batch_size", str(args.batch_size),
                "--lr", str(args.lr),
                "--seed", str(seed),
                "--run_dir", args.run_dir
            ]
            print("\n>>", " ".join(cmd))
            subprocess.run(cmd, check=True)
            # load last run summary by scanning newest directory matching prefix
            prefix = f"pcc_{args.model}_N{args.N}_seed{seed}_"
            dirs = [d for d in os.listdir(args.run_dir) if d.startswith(prefix)]
            dirs.sort(key=lambda d: os.path.getmtime(os.path.join(args.run_dir, d)))
            last = os.path.join(args.run_dir, dirs[-1], "summary.json")
            with open(last, "r", encoding="utf-8") as f:
                summary = json.load(f)
            results.append({"train_size": ts, "seed": seed, "test_acc": summary["test_acc"], "best_val_acc": summary["best_val_acc"], "run": os.path.dirname(last)})

    out = os.path.join(args.run_dir, f"sweep_{args.model}_N{args.N}_{time.strftime('%Y%m%d-%H%M%S')}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)
    print("\nWrote sweep results:", out)

if __name__ == "__main__":
    main()
