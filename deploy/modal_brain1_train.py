from __future__ import annotations

import json
import os

import modal


APP_NAME = "brain-1-train"
REMOTE_REPO = "/root/brain-1"
REMOTE_DATA = "/mnt/brain1-data"

app = modal.App(APP_NAME)

data_volume = modal.Volume.from_name("brain-1-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .add_local_dir("/Users/advaitpaliwal/Companion/Code/brain-1/src", remote_path=f"{REMOTE_REPO}/src", copy=True)
    .add_local_dir("/Users/advaitpaliwal/Companion/Code/brain-1/scripts", remote_path=f"{REMOTE_REPO}/scripts", copy=True)
    .add_local_dir("/Users/advaitpaliwal/Companion/Code/brain-1/configs", remote_path=f"{REMOTE_REPO}/configs", copy=True)
    .add_local_file("/Users/advaitpaliwal/Companion/Code/brain-1/pyproject.toml", remote_path=f"{REMOTE_REPO}/pyproject.toml", copy=True)
    .add_local_file("/Users/advaitpaliwal/Companion/Code/brain-1/README.md", remote_path=f"{REMOTE_REPO}/README.md", copy=True)
    .run_commands(f"cd {REMOTE_REPO} && pip install -e .")
)


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60 * 6,
    volumes={REMOTE_DATA: data_volume},
)
def run_text_benchmark() -> dict[str, str]:
    import subprocess

    model_config = f"{REMOTE_REPO}/configs/model.yaml"
    train_config = f"{REMOTE_REPO}/configs/algonauts_text_train_s1_s5_all4_tuned_b2_long_modal.yaml"
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REMOTE_REPO}/src"
    bundle = f"{REMOTE_DATA}/bundles/brain1_text_bundle.tar"
    part_prefix = f"{REMOTE_DATA}/bundles/brain1_text_bundle.part."
    processed_dir = f"{REMOTE_DATA}/processed/algonauts2025_text_shared_s1_sub01_sub02"
    manifests_dir = f"{REMOTE_DATA}/modal_manifests"
    canonical_manifests_dir = f"{REMOTE_DATA}/manifests"
    if not os.path.exists(bundle) and os.path.exists(f"{part_prefix}00"):
        subprocess.run(
            [
                "bash",
                "-lc",
                f"cat {part_prefix}* > {bundle}",
            ],
            check=True,
            env=env,
        )
    if os.path.exists(bundle) and not os.path.exists(processed_dir):
        subprocess.run(["mkdir", "-p", f"{REMOTE_DATA}/processed", manifests_dir], check=True, env=env)
        subprocess.run(["tar", "-xf", bundle, "-C", REMOTE_DATA], check=True, env=env)
    subprocess.run(["mkdir", "-p", canonical_manifests_dir], check=True, env=env)
    for name in [
        "algonauts2025_text_train_s1_s5_all4.modal.jsonl",
        "algonauts2025_text_val_s6_all4.modal.jsonl",
    ]:
        source = f"{manifests_dir}/{name}"
        target = f"{canonical_manifests_dir}/{name}"
        if os.path.exists(source) and not os.path.exists(target):
            subprocess.run(["cp", source, target], check=True, env=env)
    subprocess.run(
        ["python", f"{REMOTE_REPO}/scripts/train.py", "--train-config", train_config, "--model-config", model_config],
        check=True,
        env=env,
    )
    return {
        "output_dir": f"{REMOTE_DATA}/artifacts/modal_algonauts_text_s1_s5_all4_tuned_b2_long",
        "train_manifest": f"{REMOTE_DATA}/manifests/algonauts2025_text_train_s1_s5_all4.modal.jsonl",
        "val_manifest": f"{REMOTE_DATA}/manifests/algonauts2025_text_val_s6_all4.modal.jsonl",
    }


@app.local_entrypoint()
def main():
    result = run_text_benchmark.remote()
    print(json.dumps(result, indent=2))
