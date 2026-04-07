from __future__ import annotations

import json
import os

import modal


APP_NAME = "brain-1-train"
REMOTE_REPO = "/root/brain-1"
REMOTE_DATA = "/mnt/brain1-data"

app = modal.App(APP_NAME)

data_volume = modal.Volume.from_name("brain-1-data", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git", "git-annex")
    .pip_install("datalad")
    .add_local_dir("/Users/advaitpaliwal/Companion/Code/brain-1/src", remote_path=f"{REMOTE_REPO}/src", copy=True)
    .add_local_dir("/Users/advaitpaliwal/Companion/Code/brain-1/scripts", remote_path=f"{REMOTE_REPO}/scripts", copy=True)
    .add_local_dir("/Users/advaitpaliwal/Companion/Code/brain-1/configs", remote_path=f"{REMOTE_REPO}/configs", copy=True)
    .add_local_file("/Users/advaitpaliwal/Companion/Code/brain-1/pyproject.toml", remote_path=f"{REMOTE_REPO}/pyproject.toml", copy=True)
    .add_local_file("/Users/advaitpaliwal/Companion/Code/brain-1/README.md", remote_path=f"{REMOTE_REPO}/README.md", copy=True)
    .run_commands('git config --global user.email "brain1@modal.local"')
    .run_commands('git config --global user.name "brain-1 modal"')
    .run_commands(f"cd {REMOTE_REPO} && pip install -e .")
)


@app.function(
    image=image,
    timeout=60 * 60 * 6,
    volumes={REMOTE_DATA: data_volume},
    secrets=[hf_secret],
)
def download_openneuro_dataset(
    dataset: str,
    subjects_csv: str = "",
    stories_csv: str = "",
    splits_csv: str = "train,test",
    hemis_csv: str = "left,right",
    materialize_textgrids: bool = False,
    materialize_stimuli: bool = False,
    materialize_preprocessed_data: bool = False,
    materialize_metadata: bool = False,
    materialize_prepared_betas: bool = False,
) -> dict[str, str]:
    import subprocess
    import shutil
    from pathlib import Path

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REMOTE_REPO}/src"
    tmp_output = f"/tmp/openneuro-{dataset}"
    output = f"{REMOTE_DATA}/raw/openneuro/{dataset}"

    cmd = [
        "python",
        f"{REMOTE_REPO}/scripts/download_openneuro.py",
        "--dataset",
        dataset,
        "--output",
        tmp_output,
        "--backend",
        "git-annex",
    ]
    if subjects_csv:
        cmd.extend(["--subjects", subjects_csv])
    if stories_csv:
        cmd.extend(["--stories", stories_csv])
    if splits_csv:
        cmd.extend(["--splits", splits_csv])
    if hemis_csv:
        cmd.extend(["--hemis", hemis_csv])
    if materialize_textgrids:
        cmd.append("--materialize-textgrids")
    if materialize_stimuli:
        cmd.append("--materialize-stimuli")
    if materialize_preprocessed_data:
        cmd.append("--materialize-preprocessed-data")
    if materialize_metadata:
        cmd.append("--materialize-metadata")
    if materialize_prepared_betas:
        cmd.append("--materialize-prepared-betas")

    subprocess.run(cmd, check=True, env=env)

    src_root = Path(tmp_output)
    dst_root = Path(output)
    if dst_root.exists():
        shutil.rmtree(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    for path in src_root.rglob("*"):
        rel = path.relative_to(src_root)
        if any(part in {".git", ".datalad"} for part in rel.parts):
            continue
        if path.is_dir():
            (dst_root / rel).mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            target = dst_root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)

    data_volume.commit()
    return {"dataset": dataset, "output": output}


@app.function(
    image=image,
    timeout=60 * 60 * 8,
    volumes={REMOTE_DATA: data_volume},
    secrets=[hf_secret],
)
def download_git_annex_repo(
    repo_url: str,
    output_subdir: str,
    paths_csv: str = "",
) -> dict[str, str]:
    import subprocess
    import shutil
    from pathlib import Path

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REMOTE_REPO}/src"
    tmp_output = f"/tmp/{Path(output_subdir).name}"
    final_output = f"{REMOTE_DATA}/{output_subdir.strip('/')}"

    subprocess.run(
        [
            "python",
            f"{REMOTE_REPO}/scripts/download_git_annex_repo.py",
            "--repo-url",
            repo_url,
            "--output",
            tmp_output,
            "--paths",
            paths_csv,
        ],
        check=True,
        env=env,
    )

    src_root = Path(tmp_output)
    dst_root = Path(final_output)
    if dst_root.exists():
        shutil.rmtree(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    for path in src_root.rglob("*"):
        rel = path.relative_to(src_root)
        if any(part in {".git", ".datalad"} for part in rel.parts):
            continue
        if path.is_dir():
            (dst_root / rel).mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            target = dst_root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)

    data_volume.commit()
    return {"output": final_output}


@app.function(
    image=image,
    timeout=60 * 60,
    volumes={REMOTE_DATA: data_volume},
    secrets=[hf_secret],
)
def build_algonauts_av_manifest_remote(
    text_manifest: str,
    video_root: str,
    output_manifest: str,
    skip_missing_videos: bool = True,
) -> dict[str, str]:
    import subprocess

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REMOTE_REPO}/src"
    cmd = [
        "python",
        f"{REMOTE_REPO}/scripts/build_algonauts_av_manifest_from_text.py",
        "--text-manifest",
        text_manifest,
        "--video-root",
        video_root,
        "--output",
        output_manifest,
    ]
    if skip_missing_videos:
        cmd.append("--skip-missing-videos")
    subprocess.run(cmd, check=True, env=env)
    data_volume.commit()
    return {"output_manifest": output_manifest}


@app.function(
    image=image,
    timeout=60 * 60 * 8,
    volumes={REMOTE_DATA: data_volume},
    secrets=[hf_secret],
)
def sample_algonauts_video_frames_remote(
    raw_manifest: str,
    frames_root: str,
    output_manifest: str,
    max_frames: int = 0,
) -> dict[str, str]:
    import subprocess

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REMOTE_REPO}/src"
    cmd = [
        "python",
        f"{REMOTE_REPO}/scripts/sample_algonauts_video_frames.py",
        "--raw-manifest",
        raw_manifest,
        "--frames-root",
        frames_root,
        "--output-manifest",
        output_manifest,
    ]
    if max_frames:
        cmd.extend(["--max-frames", str(max_frames)])
    subprocess.run(cmd, check=True, env=env)
    data_volume.commit()
    return {"output_manifest": output_manifest, "frames_root": frames_root}


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60 * 8,
    volumes={REMOTE_DATA: data_volume},
    secrets=[hf_secret],
)
def extract_algonauts_image_features_remote(
    frame_manifest: str,
    output_manifest: str,
    feature_root: str,
    model_id: str = "google/siglip-base-patch16-224",
    batch_size: int = 16,
    device: str = "cuda",
    share_features_across_subjects: bool = True,
) -> dict[str, str]:
    import subprocess

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REMOTE_REPO}/src"
    cmd = [
        "python",
        f"{REMOTE_REPO}/scripts/extract_algonauts_image_features.py",
        "--frame-manifest",
        frame_manifest,
        "--output-manifest",
        output_manifest,
        "--feature-root",
        feature_root,
        "--model-id",
        model_id,
        "--batch-size",
        str(batch_size),
        "--device",
        device,
    ]
    if share_features_across_subjects:
        cmd.append("--share-features-across-subjects")
    subprocess.run(cmd, check=True, env=env)
    data_volume.commit()
    return {"output_manifest": output_manifest, "feature_root": feature_root}


@app.function(
    image=image,
    timeout=60 * 60 * 8,
    volumes={REMOTE_DATA: data_volume},
    secrets=[hf_secret],
)
def extract_algonauts_audio_clips_remote(
    raw_manifest: str,
    clips_root: str,
    output_manifest: str,
    max_clips: int = 0,
) -> dict[str, str]:
    import subprocess

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REMOTE_REPO}/src"
    cmd = [
        "python",
        f"{REMOTE_REPO}/scripts/extract_algonauts_audio_clips.py",
        "--raw-manifest",
        raw_manifest,
        "--clips-root",
        clips_root,
        "--output-manifest",
        output_manifest,
    ]
    if max_clips:
        cmd.extend(["--max-clips", str(max_clips)])
    subprocess.run(cmd, check=True, env=env)
    data_volume.commit()
    return {"output_manifest": output_manifest, "clips_root": clips_root}


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60 * 8,
    volumes={REMOTE_DATA: data_volume},
    secrets=[hf_secret],
)
def extract_algonauts_audio_features_remote(
    clip_manifest: str,
    output_manifest: str,
    feature_root: str,
    model_id: str = "facebook/w2v-bert-2.0",
    batch_size: int = 8,
    device: str = "cuda",
    share_features_across_subjects: bool = True,
) -> dict[str, str]:
    import subprocess

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REMOTE_REPO}/src"
    cmd = [
        "python",
        f"{REMOTE_REPO}/scripts/extract_algonauts_audio_features.py",
        "--clip-manifest",
        clip_manifest,
        "--output-manifest",
        output_manifest,
        "--feature-root",
        feature_root,
        "--model-id",
        model_id,
        "--batch-size",
        str(batch_size),
        "--device",
        device,
    ]
    if share_features_across_subjects:
        cmd.append("--share-features-across-subjects")
    subprocess.run(cmd, check=True, env=env)
    data_volume.commit()
    return {"output_manifest": output_manifest, "feature_root": feature_root}


@app.function(
    image=image,
    timeout=60 * 30,
    volumes={REMOTE_DATA: data_volume},
    secrets=[hf_secret],
)
def build_trimodal_manifest_remote(
    text_manifest: str,
    audio_manifest: str,
    video_manifest: str,
    output_manifest: str,
) -> dict[str, str]:
    import subprocess

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REMOTE_REPO}/src"
    subprocess.run(
        [
            "python",
            f"{REMOTE_REPO}/scripts/build_trimodal_manifest.py",
            "--text-manifest",
            text_manifest,
            "--audio-manifest",
            audio_manifest,
            "--video-manifest",
            video_manifest,
            "--output",
            output_manifest,
        ],
        check=True,
        env=env,
    )
    data_volume.commit()
    return {"output_manifest": output_manifest}


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60 * 8,
    volumes={REMOTE_DATA: data_volume},
    secrets=[hf_secret],
)
def run_trimodal_benchmark(
    train_manifest: str,
    val_manifest: str,
    output_dir: str,
    max_steps: int = 200,
    lr: float = 1.0e-4,
    eval_every: int = 25,
) -> dict[str, str]:
    import subprocess
    from pathlib import Path
    import yaml

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REMOTE_REPO}/src"
    config_path = f"{REMOTE_REPO}/configs/trimodal_modal.yaml"

    config_payload = {
        "data": {
            "manifest_path": train_manifest,
            "val_manifest_path": val_manifest,
        },
        "optimization": {
            "batch_size": 2,
            "lr": lr,
            "weight_decay": 0.01,
            "max_steps": max_steps,
            "warmup_steps": 0,
        },
        "loss": {
            "mse_weight": 1.0,
            "correlation_weight": 0.0,
        },
        "training": {
            "device": "cuda",
            "log_every": 10,
            "eval_every": eval_every,
            "grad_clip_norm": 1.0,
            "output_dir": output_dir,
        },
    }

    Path(config_path).write_text(yaml.safe_dump(config_payload), encoding="utf-8")
    subprocess.run(
        [
            "python",
            f"{REMOTE_REPO}/scripts/train_trimodal.py",
            "--config",
            config_path,
        ],
        check=True,
        env=env,
    )
    return {"output_dir": output_dir}


@app.function(
    image=image,
    timeout=60 * 60,
    volumes={REMOTE_DATA: data_volume},
    secrets=[hf_secret],
)
def build_lebel2023_text_manifest_remote(
    subjects_csv: str,
    stories_csv: str,
    output_manifest: str = f"{REMOTE_DATA}/manifests/lebel2023_text_raw.jsonl",
) -> dict[str, str]:
    import subprocess

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REMOTE_REPO}/src"
    dataset_root = f"{REMOTE_DATA}/raw/openneuro/ds003020"
    subprocess.run(
        [
            "python",
            f"{REMOTE_REPO}/scripts/build_lebel2023_text_manifest.py",
            "--dataset-root",
            dataset_root,
            "--subjects",
            subjects_csv,
            "--stories",
            stories_csv,
            "--output",
            output_manifest,
        ],
        check=True,
        env=env,
    )
    data_volume.commit()
    return {"output_manifest": output_manifest}


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60 * 6,
    volumes={REMOTE_DATA: data_volume},
    secrets=[hf_secret],
)
def extract_text_features_remote(
    raw_manifest: str,
    output_manifest: str,
    feature_root: str,
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
    batch_size: int = 32,
    device: str = "cuda",
    share_features_across_subjects: bool = True,
) -> dict[str, str]:
    import subprocess

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REMOTE_REPO}/src"
    cmd = [
        "python",
        f"{REMOTE_REPO}/scripts/extract_algonauts_text_features.py",
        "--raw-manifest",
        raw_manifest,
        "--output-manifest",
        output_manifest,
        "--feature-root",
        feature_root,
        "--model-id",
        model_id,
        "--batch-size",
        str(batch_size),
        "--device",
        device,
    ]
    if share_features_across_subjects:
        cmd.append("--share-features-across-subjects")
    subprocess.run(cmd, check=True, env=env)
    data_volume.commit()
    return {"output_manifest": output_manifest, "feature_root": feature_root}


@app.function(
    image=image,
    timeout=60 * 60,
    volumes={REMOTE_DATA: data_volume},
    secrets=[hf_secret],
)
def build_bold_moments_text_manifest_remote(
    subject: str,
    split: str = "train",
    hemi: str = "left",
    limit: int = 0,
    output_manifest: str = f"{REMOTE_DATA}/manifests/bold_moments_text_raw.jsonl",
    target_root: str = f"{REMOTE_DATA}/processed/bold_moments_targets",
) -> dict[str, str]:
    import subprocess

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REMOTE_REPO}/src"
    dataset_root = f"{REMOTE_DATA}/raw/openneuro/ds005165"
    cmd = [
        "python",
        f"{REMOTE_REPO}/scripts/build_bold_moments_text_manifest.py",
        "--dataset-root",
        dataset_root,
        "--subject",
        subject,
        "--split",
        split,
        "--hemi",
        hemi,
        "--target-root",
        target_root,
        "--output",
        output_manifest,
    ]
    if limit:
        cmd.extend(["--limit", str(limit)])
    subprocess.run(cmd, check=True, env=env)
    data_volume.commit()
    return {"output_manifest": output_manifest, "target_root": target_root}


@app.function(
    image=image,
    timeout=60 * 30,
    volumes={REMOTE_DATA: data_volume},
    secrets=[hf_secret],
)
def concat_manifests_remote(
    manifests_csv: str,
    output_manifest: str,
    starting_index: int = 0,
) -> dict[str, str]:
    import subprocess

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REMOTE_REPO}/src"
    manifests = [value.strip() for value in manifests_csv.split(",") if value.strip()]
    cmd = [
        "python",
        f"{REMOTE_REPO}/scripts/concat_manifests.py",
        "--output",
        output_manifest,
        "--starting-index",
        str(starting_index),
        "--manifests",
        *manifests,
    ]
    subprocess.run(cmd, check=True, env=env)
    data_volume.commit()
    return {"output_manifest": output_manifest}


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60 * 8,
    volumes={REMOTE_DATA: data_volume},
    secrets=[hf_secret],
)
def run_multidataset_text_benchmark(
    train_manifest: str,
    output_dir: str,
    val_manifest: str = "",
    model_config_path: str = f"{REMOTE_REPO}/configs/model.yaml",
    eval_every: int = 25,
    max_steps: int = 300,
    init_checkpoint: str = "",
    algonauts_weight: float = 1.0,
    lebel_weight: float = 1.0,
    bold_weight: float = 1.0,
) -> dict[str, str]:
    import subprocess
    from pathlib import Path
    import yaml

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REMOTE_REPO}/src"
    train_config_path = f"{REMOTE_REPO}/configs/multidataset_text_train_modal.yaml"

    config_payload = {
        "seed": 1337,
        "data": {
            "manifest_path": train_manifest,
            "val_manifest_path": val_manifest if val_manifest else None,
            "synthetic_fallback_size": 0,
            "synthetic_seq_len": 0,
            "dataset_weights": {
                "algonauts2025": algonauts_weight,
                "lebel2023": lebel_weight,
                "bold_moments_left": bold_weight,
            },
        },
        "optimization": {
            "batch_size": 2,
            "lr": 1.0e-4,
            "weight_decay": 0.01,
            "max_steps": max_steps,
            "warmup_steps": 0,
        },
        "loss": {
            "mse_weight": 1.0,
            "correlation_weight": 0.0,
        },
        "training": {
            "device": "cuda",
            "mixed_precision": "no",
            "grad_clip_norm": 1.0,
            "log_every": 10,
            "eval_every": eval_every if val_manifest else 0,
            "checkpoint_every": 100,
            "output_dir": output_dir,
            "init_checkpoint_path": init_checkpoint if init_checkpoint else None,
        },
    }

    Path(train_config_path).write_text(yaml.safe_dump(config_payload), encoding="utf-8")
    subprocess.run(
        [
            "python",
            f"{REMOTE_REPO}/scripts/train_multidataset.py",
            "--train-config",
            train_config_path,
            "--model-config",
            model_config_path,
        ],
        check=True,
        env=env,
    )
    return {"output_dir": output_dir, "train_manifest": train_manifest}


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60 * 6,
    volumes={REMOTE_DATA: data_volume},
    secrets=[hf_secret],
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


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60 * 8,
    volumes={REMOTE_DATA: data_volume},
    secrets=[hf_secret],
)
def run_qwen25_3b_text_extraction() -> dict[str, str]:
    import subprocess

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REMOTE_REPO}/src"
    chunk_manifest = f"{REMOTE_DATA}/manifests/algonauts2025_qwen25_3b_text_chunks_s1_s6_all4.jsonl"
    output_manifest = f"{REMOTE_DATA}/manifests/algonauts2025_qwen25_3b_text_train_s1_s6_all4.modal.jsonl"
    feature_root = f"{REMOTE_DATA}/processed/algonauts2025_qwen25_3b_text_shared"

    subprocess.run(
        [
            "python",
            f"{REMOTE_REPO}/scripts/extract_algonauts_text_features.py",
            "--raw-manifest",
            chunk_manifest,
            "--output-manifest",
            output_manifest,
            "--feature-root",
            feature_root,
            "--batch-size",
            "16",
            "--device",
            "cuda",
            "--model-id",
            "Qwen/Qwen2.5-3B-Instruct",
            "--share-features-across-subjects",
        ],
        check=True,
        env=env,
    )
    return {
        "output_manifest": output_manifest,
        "feature_root": feature_root,
    }


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60 * 8,
    volumes={REMOTE_DATA: data_volume},
    secrets=[hf_secret],
)
def run_qwen25_3b_text_benchmark() -> dict[str, str]:
    import subprocess

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REMOTE_REPO}/src"
    train_manifest = f"{REMOTE_DATA}/manifests/algonauts2025_qwen25_3b_text_train_s1_s5_all4.modal.jsonl"
    val_manifest = f"{REMOTE_DATA}/manifests/algonauts2025_qwen25_3b_text_val_s6_all4.modal.jsonl"
    output_dir = f"{REMOTE_DATA}/artifacts/modal_algonauts_qwen25_3b_text_s1_s5_all4"
    train_config_path = f"{REMOTE_REPO}/configs/algonauts_qwen25_3b_text_s1_s5_all4_modal.yaml"
    model_config_path = f"{REMOTE_REPO}/configs/model.yaml"

    config_payload = {
        "seed": 1337,
        "data": {
            "manifest_path": train_manifest,
            "val_manifest_path": val_manifest,
            "synthetic_fallback_size": 0,
            "synthetic_seq_len": 0,
        },
        "optimization": {
            "batch_size": 2,
            "lr": 1.0e-4,
            "weight_decay": 0.01,
            "max_steps": 600,
            "warmup_steps": 0,
        },
        "loss": {
            "mse_weight": 1.0,
            "correlation_weight": 0.0,
        },
        "training": {
            "device": "cuda",
            "mixed_precision": "no",
            "grad_clip_norm": 1.0,
            "log_every": 10,
            "eval_every": 25,
            "checkpoint_every": 100,
            "output_dir": output_dir,
        },
    }

    Path = __import__("pathlib").Path
    Path(train_config_path).write_text(__import__("yaml").safe_dump(config_payload), encoding="utf-8")
    subprocess.run(
        ["python", f"{REMOTE_REPO}/scripts/train.py", "--train-config", train_config_path, "--model-config", model_config_path],
        check=True,
        env=env,
    )
    return {
        "output_dir": output_dir,
        "train_manifest": train_manifest,
        "val_manifest": val_manifest,
    }


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60 * 8,
    volumes={REMOTE_DATA: data_volume},
    secrets=[hf_secret],
)
def run_text_benchmark_wide() -> dict[str, str]:
    import subprocess
    from pathlib import Path
    import yaml

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REMOTE_REPO}/src"
    train_manifest = f"{REMOTE_DATA}/manifests/algonauts2025_text_train_s1_s5_all4.modal.jsonl"
    val_manifest = f"{REMOTE_DATA}/manifests/algonauts2025_text_val_s6_all4.modal.jsonl"
    output_dir = f"{REMOTE_DATA}/artifacts/modal_algonauts_text_s1_s5_all4_wide"
    train_config_path = f"{REMOTE_REPO}/configs/algonauts_text_train_s1_s5_all4_wide_modal.yaml"
    model_config_path = f"{REMOTE_REPO}/configs/model_text_wide.yaml"

    config_payload = {
        "seed": 1337,
        "data": {
            "manifest_path": train_manifest,
            "val_manifest_path": val_manifest,
            "synthetic_fallback_size": 0,
            "synthetic_seq_len": 0,
        },
        "optimization": {
            "batch_size": 2,
            "lr": 1.0e-4,
            "weight_decay": 0.01,
            "max_steps": 800,
            "warmup_steps": 0,
        },
        "loss": {
            "mse_weight": 1.0,
            "correlation_weight": 0.0,
        },
        "training": {
            "device": "cuda",
            "mixed_precision": "no",
            "grad_clip_norm": 1.0,
            "log_every": 10,
            "eval_every": 25,
            "checkpoint_every": 100,
            "output_dir": output_dir,
        },
    }

    Path(train_config_path).write_text(yaml.safe_dump(config_payload), encoding="utf-8")
    subprocess.run(
        ["python", f"{REMOTE_REPO}/scripts/train.py", "--train-config", train_config_path, "--model-config", model_config_path],
        check=True,
        env=env,
    )
    return {
        "output_dir": output_dir,
        "train_manifest": train_manifest,
        "val_manifest": val_manifest,
    }


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60 * 8,
    volumes={REMOTE_DATA: data_volume},
    secrets=[hf_secret],
)
def run_single_dataset_benchmark(
    train_manifest: str,
    output_dir: str,
    val_manifest: str = "",
    init_checkpoint: str = "",
    model_config_path: str = f"{REMOTE_REPO}/configs/model.yaml",
    eval_every: int = 25,
    max_steps: int = 300,
    lr: float = 1.0e-4,
) -> dict[str, str]:
    import subprocess
    from pathlib import Path
    import yaml

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REMOTE_REPO}/src"
    train_config_path = f"{REMOTE_REPO}/configs/single_dataset_modal.yaml"

    config_payload = {
        "seed": 1337,
        "data": {
            "manifest_path": train_manifest,
            "val_manifest_path": val_manifest if val_manifest else None,
            "synthetic_fallback_size": 0,
            "synthetic_seq_len": 0,
        },
        "optimization": {
            "batch_size": 2,
            "lr": lr,
            "weight_decay": 0.01,
            "max_steps": max_steps,
            "warmup_steps": 0,
        },
        "loss": {
            "mse_weight": 1.0,
            "correlation_weight": 0.0,
        },
        "training": {
            "device": "cuda",
            "mixed_precision": "no",
            "grad_clip_norm": 1.0,
            "log_every": 10,
            "eval_every": eval_every if val_manifest else 0,
            "checkpoint_every": 100,
            "output_dir": output_dir,
            "init_checkpoint_path": init_checkpoint if init_checkpoint else None,
        },
    }

    Path(train_config_path).write_text(yaml.safe_dump(config_payload), encoding="utf-8")
    subprocess.run(
        [
            "python",
            f"{REMOTE_REPO}/scripts/train.py",
            "--train-config",
            train_config_path,
            "--model-config",
            model_config_path,
        ],
        check=True,
        env=env,
    )
    return {"output_dir": output_dir, "train_manifest": train_manifest}


@app.local_entrypoint()
def main():
    result = run_text_benchmark.remote()
    print(json.dumps(result, indent=2))
