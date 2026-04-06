from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from urllib.request import urlopen
from pathlib import Path


def parse_csv(value: str) -> list[str]:
    return [chunk.strip() for chunk in value.split(",") if chunk.strip()]


def run(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=str(cwd) if cwd is not None else None, check=True)


def download_http(url: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        print(f"Using existing file: {output}")
        return
    print(f"Downloading {url} -> {output}")
    with urlopen(url) as response, output.open("wb") as handle:
        shutil.copyfileobj(response, handle, length=1024 * 1024)


def install_repo_datalad(dataset: str, output: Path) -> None:
    if (output / ".git").exists():
        print(f"Using existing dataset checkout at {output}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    source = f"https://github.com/OpenNeuroDatasets/{dataset}.git"
    print(f"Installing {dataset} from {source} into {output}")
    run(
        [
            sys.executable,
            "-c",
            (
                "import datalad.api as dl; "
                f"dl.install(path={str(output)!r}, source={source!r})"
            ),
        ]
    )


def install_repo_git_annex(dataset: str, output: Path) -> None:
    if (output / ".git").exists():
        print(f"Using existing dataset checkout at {output}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    source = f"https://github.com/OpenNeuroDatasets/{dataset}.git"
    print(f"Cloning {dataset} from {source} into {output}")
    run(["git", "clone", source, str(output)])
    run(["git", "config", "user.email", "brain1@local.invalid"], cwd=output)
    run(["git", "config", "user.name", "brain-1"], cwd=output)
    run(["git", "annex", "init", "brain-1"], cwd=output)


def install_repo(dataset: str, output: Path, backend: str) -> None:
    if backend == "datalad":
        install_repo_datalad(dataset, output)
        return
    if backend == "git-annex":
        install_repo_git_annex(dataset, output)
        return
    raise ValueError(f"Unsupported backend: {backend}")


def build_ds003020_paths(
    subjects: list[str],
    stories: list[str],
    include_textgrids: bool,
    include_stimuli: bool,
    include_preprocessed_data: bool,
) -> list[str]:
    paths: list[str] = []
    if include_textgrids:
        if stories:
            paths.extend([f"derivatives/TextGrids/{story}.TextGrid" for story in stories])
        else:
            paths.append("derivatives/TextGrids")
    if include_stimuli:
        if stories:
            paths.extend([f"stimuli/{story}.wav" for story in stories])
        else:
            paths.append("stimuli")
    if include_preprocessed_data:
        if subjects and stories:
            for subject in subjects:
                for story in stories:
                    paths.append(f"derivatives/preprocessed_data/{subject}/{story}.hf5")
        elif subjects:
            paths.extend([f"derivatives/preprocessed_data/{subject}" for subject in subjects])
        else:
            paths.append("derivatives/preprocessed_data")
    return paths


def build_ds005165_paths(
    subjects: list[str],
    include_metadata: bool,
    include_prepared_betas: bool,
    splits: list[str],
    hemis: list[str],
) -> list[str]:
    paths: list[str] = []
    if include_metadata:
        paths.extend(
            [
                "derivatives/stimuli_metadata/llm_frame_annotations.json",
                "derivatives/stimuli_metadata/annotations_fieldnames.json",
                "derivatives/stimuli_metadata/README.txt",
            ]
        )
    if include_prepared_betas:
        if subjects:
            for subject in subjects:
                for split in splits:
                    for hemi in hemis:
                        paths.append(
                            f"derivatives/versionB/fsaverage/GLM/{subject}/prepared_betas/"
                            f"{subject}_organized_betas_task-{split}_hemi-{hemi}_normalized.pkl"
                        )
        else:
            paths.append("derivatives/versionB/fsaverage/GLM")
    return paths


def maybe_download_direct(dataset: str, output: Path, paths: list[str]) -> bool:
    if dataset != "ds005165":
        return False
    prepared_beta_prefix = "derivatives/versionB/fsaverage/GLM/"
    prepared_beta_paths = [
        rel_path
        for rel_path in paths
        if rel_path.startswith(prepared_beta_prefix)
        and rel_path.endswith("_normalized.pkl")
    ]
    if not prepared_beta_paths:
        return False
    for rel_path in prepared_beta_paths:
        url = f"https://s3.amazonaws.com/openneuro.org/{dataset}/{rel_path}"
        download_http(url, output / rel_path)
    return len(prepared_beta_paths) == len(paths)


def main() -> None:
    parser = argparse.ArgumentParser(description="Install and selectively materialize OpenNeuro datasets.")
    parser.add_argument("--dataset", required=True, help="OpenNeuro dataset id, e.g. ds003020")
    parser.add_argument("--output", required=True, help="Destination directory")
    parser.add_argument(
        "--subjects",
        default="",
        help="Optional comma-separated subject ids, e.g. UTS01,UTS02 or sub-01,sub-02",
    )
    parser.add_argument(
        "--stories",
        default="",
        help="Optional comma-separated story ids for ds003020, e.g. wheretheressmoke,legacy",
    )
    parser.add_argument(
        "--materialize-textgrids",
        action="store_true",
        help="For ds003020, download TextGrid alignments",
    )
    parser.add_argument(
        "--materialize-stimuli",
        action="store_true",
        help="For ds003020, download WAV stimulus files",
    )
    parser.add_argument(
        "--materialize-preprocessed-data",
        action="store_true",
        help="For ds003020, download HDF5 response files",
    )
    parser.add_argument(
        "--materialize-metadata",
        action="store_true",
        help="For ds005165, download derivative stimulus metadata",
    )
    parser.add_argument(
        "--materialize-prepared-betas",
        action="store_true",
        help="For ds005165, download versionB prepared beta files",
    )
    parser.add_argument(
        "--backend",
        default="datalad",
        choices=["datalad", "git-annex"],
        help="Repository setup backend to use before git-annex get",
    )
    parser.add_argument(
        "--splits",
        default="train,test",
        help="For ds005165 prepared betas, comma-separated splits to fetch",
    )
    parser.add_argument(
        "--hemis",
        default="left,right",
        help="For ds005165 prepared betas, comma-separated hemispheres to fetch",
    )
    args = parser.parse_args()

    output = Path(args.output).expanduser().resolve()
    subjects = parse_csv(args.subjects)
    stories = parse_csv(args.stories)
    splits = parse_csv(args.splits)
    hemis = parse_csv(args.hemis)

    install_repo(args.dataset, output, backend=args.backend)

    if args.dataset == "ds003020":
        paths = build_ds003020_paths(
            subjects=subjects,
            stories=stories,
            include_textgrids=args.materialize_textgrids,
            include_stimuli=args.materialize_stimuli,
            include_preprocessed_data=args.materialize_preprocessed_data,
        )
    elif args.dataset == "ds005165":
        paths = build_ds005165_paths(
            subjects=subjects,
            include_metadata=args.materialize_metadata,
            include_prepared_betas=args.materialize_prepared_betas,
            splits=splits,
            hemis=hemis,
        )
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    if not paths:
        print(f"Installed {args.dataset} at {output}")
        print("No materialization flags were set, so only the dataset checkout was created.")
        return

    if maybe_download_direct(args.dataset, output, paths):
        print(f"Finished downloading requested data for {args.dataset} into {output}")
        return

    print(f"Materializing {len(paths)} path(s) in {output}")
    run(["git", "annex", "get", *paths], cwd=output)
    print(f"Finished downloading requested data for {args.dataset} into {output}")


if __name__ == "__main__":
    main()
