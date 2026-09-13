"""
synthetic_data.py
------------------
Generates a realistic, labelled, CIC-IDS-2018-style flow-level + packet-level
traffic dataset entirely offline. This lets Zynex's full pipeline (feature
extraction -> world model training -> K-step prediction -> demo UI) be
demonstrated to judges without needing to download a multi-GB real dataset.

The generator simulates a benign traffic baseline interrupted by several
full attack campaigns, each unfolding through the MITRE ATT&CK stages:

    Benign -> Reconnaissance -> Initial Access -> Lateral Movement
           -> Command & Control -> Exfiltration -> (back to) Benign

Every row is one NetFlow-style flow record with an explicit timestamp,
so `feature_extraction.py` can bucket rows into time windows and derive
ground-truth state-transition + MITRE-stage labels, exactly as it would
for a real CIC-IDS-2018 / CTU-13 CSV.

Usage:
    python src/synthetic_data.py --out data/synthetic_traffic.csv --n_windows 4000
"""
import argparse
import numpy as np
import pandas as pd

STAGES = ["Benign", "Reconnaissance", "Initial Access", "Lateral Movement",
          "Command & Control", "Exfiltration"]

RNG_SEED = 42


def _rand_ip(rng, subnet="10.0.0."):
    return f"{subnet}{rng.integers(2, 254)}"


def _gen_benign_flow(rng, t):
    return dict(
        timestamp=t,
        src_ip=_rand_ip(rng), dst_ip=_rand_ip(rng),
        src_port=rng.integers(1024, 65535), dst_port=rng.choice([80, 443, 53, 22, 3306]),
        protocol=rng.choice(["TCP", "UDP"]),
        syn=rng.integers(0, 2), ack=1, fin=rng.integers(0, 2), rst=0, psh=rng.integers(0, 2), urg=0,
        bytes=max(50, int(rng.normal(1500, 500))),
        packets=max(1, int(rng.normal(10, 4))),
        duration=abs(rng.normal(0.5, 0.3)),
        iat_mean=abs(rng.normal(0.05, 0.02)), iat_var=abs(rng.normal(0.01, 0.005)), iat_max=abs(rng.normal(0.2, 0.1)),
        bidir_ratio=rng.uniform(0.4, 0.9),
        ttl_mean=64, ttl_var=abs(rng.normal(0.5, 0.2)),
        win_size=rng.integers(4096, 65536),
        frag_flag=0,
        payload_mean=max(0, rng.normal(800, 300)), payload_skew=rng.normal(0, 0.5),
        scan_score=0.0,
        retrans=rng.integers(0, 2),
        label="Benign", stage="Benign",
    )


def _gen_recon_flow(rng, t, target_ip):
    # sequential port sweep against one target -> low bytes, many distinct ports, low IAT
    return dict(
        timestamp=t,
        src_ip="10.0.0.199", dst_ip=target_ip,
        src_port=rng.integers(1024, 65535), dst_port=int(rng.integers(1, 1024)),
        protocol="TCP",
        syn=1, ack=0, fin=0, rst=rng.integers(0, 2), psh=0, urg=0,
        bytes=rng.integers(40, 80),
        packets=1,
        duration=abs(rng.normal(0.01, 0.005)),
        iat_mean=abs(rng.normal(0.003, 0.001)), iat_var=abs(rng.normal(0.0005, 0.0002)), iat_max=abs(rng.normal(0.01, 0.003)),
        bidir_ratio=rng.uniform(0.0, 0.2),
        ttl_mean=64, ttl_var=abs(rng.normal(0.1, 0.05)),
        win_size=rng.integers(512, 2048),
        frag_flag=0,
        payload_mean=0, payload_skew=0,
        scan_score=rng.uniform(0.7, 1.0),
        retrans=0,
        label="Malicious", stage="Reconnaissance",
    )


def _gen_initial_access_flow(rng, t, target_ip):
    # brute-force / exploit attempt -> repeated auth flows to same port, some SYN floods
    return dict(
        timestamp=t,
        src_ip="10.0.0.199", dst_ip=target_ip,
        src_port=rng.integers(1024, 65535), dst_port=rng.choice([22, 3389, 445, 21]),
        protocol="TCP",
        syn=1, ack=rng.integers(0, 2), fin=0, rst=rng.integers(0, 2), psh=1, urg=0,
        bytes=rng.integers(100, 300),
        packets=rng.integers(2, 6),
        duration=abs(rng.normal(0.05, 0.02)),
        iat_mean=abs(rng.normal(0.01, 0.004)), iat_var=abs(rng.normal(0.002, 0.001)), iat_max=abs(rng.normal(0.03, 0.01)),
        bidir_ratio=rng.uniform(0.1, 0.3),
        ttl_mean=64, ttl_var=abs(rng.normal(0.3, 0.1)),
        win_size=rng.integers(1024, 4096),
        frag_flag=0,
        payload_mean=rng.normal(150, 50), payload_skew=rng.normal(0.5, 0.2),
        scan_score=rng.uniform(0.3, 0.6),
        retrans=rng.integers(0, 3),
        label="Malicious", stage="Initial Access",
    )


def _gen_lateral_flow(rng, t):
    # internal east-west traffic spike between many internal hosts
    return dict(
        timestamp=t,
        src_ip="10.0.0.199", dst_ip=_rand_ip(rng),
        src_port=rng.integers(1024, 65535), dst_port=rng.choice([445, 135, 139, 3389]),
        protocol="TCP",
        syn=1, ack=1, fin=rng.integers(0, 2), rst=0, psh=1, urg=0,
        bytes=rng.integers(500, 3000),
        packets=rng.integers(5, 15),
        duration=abs(rng.normal(0.3, 0.1)),
        iat_mean=abs(rng.normal(0.02, 0.01)), iat_var=abs(rng.normal(0.005, 0.002)), iat_max=abs(rng.normal(0.05, 0.02)),
        bidir_ratio=rng.uniform(0.5, 0.8),
        ttl_mean=64, ttl_var=abs(rng.normal(0.2, 0.1)),
        win_size=rng.integers(8192, 32768),
        frag_flag=0,
        payload_mean=rng.normal(1200, 400), payload_skew=rng.normal(0.2, 0.1),
        scan_score=rng.uniform(0.2, 0.4),
        retrans=rng.integers(0, 2),
        label="Malicious", stage="Lateral Movement",
    )


def _gen_c2_flow(rng, t):
    # periodic low-volume beaconing to a single external IP
    return dict(
        timestamp=t,
        src_ip="10.0.0.199", dst_ip="198.51.100.7",
        src_port=rng.integers(1024, 65535), dst_port=rng.choice([443, 8080, 53]),
        protocol="TCP",
        syn=1, ack=1, fin=0, rst=0, psh=1, urg=0,
        bytes=rng.integers(80, 200),
        packets=rng.integers(2, 4),
        duration=abs(rng.normal(0.02, 0.005)),
        iat_mean=abs(rng.normal(30.0, 2.0)), iat_var=abs(rng.normal(1.0, 0.3)), iat_max=abs(rng.normal(32.0, 2.0)),
        bidir_ratio=rng.uniform(0.4, 0.6),
        ttl_mean=54, ttl_var=abs(rng.normal(0.1, 0.05)),
        win_size=rng.integers(512, 2048),
        frag_flag=0,
        payload_mean=rng.normal(120, 30), payload_skew=rng.normal(0.1, 0.05),
        scan_score=0.0,
        retrans=0,
        label="Malicious", stage="Command & Control",
    )


def _gen_exfil_flow(rng, t):
    # large sustained outbound transfer
    return dict(
        timestamp=t,
        src_ip="10.0.0.199", dst_ip="198.51.100.7",
        src_port=rng.integers(1024, 65535), dst_port=443,
        protocol="TCP",
        syn=0, ack=1, fin=0, rst=0, psh=1, urg=0,
        bytes=rng.integers(50000, 200000),
        packets=rng.integers(200, 800),
        duration=abs(rng.normal(5.0, 1.5)),
        iat_mean=abs(rng.normal(0.005, 0.002)), iat_var=abs(rng.normal(0.001, 0.0005)), iat_max=abs(rng.normal(0.02, 0.01)),
        bidir_ratio=rng.uniform(0.1, 0.2),
        ttl_mean=54, ttl_var=abs(rng.normal(0.1, 0.05)),
        win_size=rng.integers(32768, 65536),
        frag_flag=rng.integers(0, 2),
        payload_mean=rng.normal(1400, 100), payload_skew=rng.normal(-0.2, 0.1),
        scan_score=0.0,
        retrans=rng.integers(0, 5),
        label="Malicious", stage="Exfiltration",
    )


STAGE_GENERATORS = {
    "Reconnaissance": _gen_recon_flow,
    "Initial Access": _gen_initial_access_flow,
    "Lateral Movement": lambda rng, t, target_ip=None: _gen_lateral_flow(rng, t),
    "Command & Control": lambda rng, t, target_ip=None: _gen_c2_flow(rng, t),
    "Exfiltration": lambda rng, t, target_ip=None: _gen_exfil_flow(rng, t),
}


def generate(n_windows=4000, flows_per_window=6, campaign_every=600, campaign_len=250, seed=RNG_SEED):
    """
    Generates a flow-level dataset spanning `n_windows` one-second time windows,
    with `flows_per_window` benign flows per window on average, and injects a
    full attack campaign (recon -> initial access -> lateral -> C2 -> exfil)
    every `campaign_every` windows, lasting `campaign_len` windows.
    """
    rng = np.random.default_rng(seed)
    rows = []
    t = 0.0
    campaign_active = False
    campaign_start = None
    target_ip = None

    for w in range(n_windows):
        # decide if a new campaign should start
        if not campaign_active and w % campaign_every == 0 and w > 0:
            campaign_active = True
            campaign_start = w
            target_ip = _rand_ip(rng)

        # background benign traffic always present
        n_benign = rng.poisson(flows_per_window)
        for _ in range(n_benign):
            rows.append(_gen_benign_flow(rng, t))
            t += rng.uniform(0.05, 0.2)

        if campaign_active:
            progress = (w - campaign_start) / campaign_len
            if progress < 0.15:
                stage = "Reconnaissance"
                n_mal = rng.integers(8, 20)
            elif progress < 0.35:
                stage = "Initial Access"
                n_mal = rng.integers(3, 10)
            elif progress < 0.65:
                stage = "Lateral Movement"
                n_mal = rng.integers(2, 6)
            elif progress < 0.85:
                stage = "Command & Control"
                n_mal = rng.integers(1, 3)
            elif progress < 1.0:
                stage = "Exfiltration"
                n_mal = rng.integers(1, 3)
            else:
                campaign_active = False
                target_ip = None
                n_mal = 0
                stage = None

            if stage is not None:
                gen_fn = STAGE_GENERATORS[stage]
                for _ in range(int(n_mal)):
                    rows.append(gen_fn(rng, t, target_ip))
                    t += rng.uniform(0.01, 0.1)

        t += 1.0  # advance to next window's base time

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    return df


def main():
    ap = argparse.ArgumentParser(description="Generate synthetic CIC-IDS-2018-style traffic for Zynex")
    ap.add_argument("--out", type=str, default="data/synthetic_traffic.csv")
    ap.add_argument("--n_windows", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=RNG_SEED)
    args = ap.parse_args()

    df = generate(n_windows=args.n_windows, seed=args.seed)
    df.to_csv(args.out, index=False)
    print(f"[synthetic_data] wrote {len(df)} flow records ({df['label'].value_counts().to_dict()}) -> {args.out}")


if __name__ == "__main__":
    main()
