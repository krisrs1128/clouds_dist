from pathlib import Path
import subprocess
import argparse

# -----------------------------------------------------------------------------
# -----  This files's purpose is to re-run all the EXISTING sbatch files  -----
# -----  with the config files that are in subdirectories (1 level),      -----
# -----  along with EXISTING configuration files                          -----
# -----  Modifications:                                                   -----
# -----    - adds `--resume` flag at the begining of the command          -----
# -----    - checks out the commit in `hash.txt`                          -----
# -----------------------------------------------------------------------------


def process_to_resume(l, git_hash, exp_id):

    # TODO validate on singularity

    new_l = l[:-1].replace("\\", "")
    new_l += " --resume"
    new_l += " --existing_exp_id={}".format(exp_id)
    new_l += " \\\n"
    new_l = "git checkout {} && ".format(git_hash) + new_l


def get_text(file_path):
    if (file_path).exists():
        with (file_path).open("r") as f:
            return f.read().strip()
    raise ValueError("{} does not exist".format(str(file_path)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--exp_dir",
        type=str,
        help="where to find the exp_dir where the experiments are stored",
    )
    parser.add_argument(
        "--test_mode",
        action="store_true",
        help="create files but not run sbatch to test",
    )
    parser.add_argument("-p", "--python_command", type=str, default="-m src.train")
    opts = parser.parse_args()

    exp_dir = Path(opts.exp_dir).resolve()

    assert exp_dir.exists(), "{} does not exist".format(exp_dir)

    # ------------------------------------
    # -----   Find experiment runs   -----
    # ------------------------------------
    runs = [
        d
        for d in exp_dir.glob("run_*")
        if d.is_dir()
        and ("checkpoints" in [e.name for e in d.iterdir()])
        and len(list((d / "checkpoints").glob("*.pt"))) > 0
    ]

    for run in runs:
        # ---------------------------------------------------
        # -----  Check for existing resume sbatch file  -----
        # ---------------------------------------------------
        resume_sbatch_files = list(run.glob("*_resume.sh"))
        assert (
            len(resume_sbatch_files) <= 1
        ), "More than 1 _resume.sh line in {}".format(run)

        if len(resume_sbatch_files) == 1:
            # -----------------------------------------------------------------
            # -----   If there's an existing resume sbatch file, use it   -----
            # -----------------------------------------------------------------
            resume_sbatch_file = resume_sbatch_files[0]
            print(
                subprocess.check_output(f"sbatch {str(resume_sbatch_file)}", shell=True)
            )
        else:
            # ----------------------------------------------------------------
            # -----  Otherwise create one by adding "--resume" to the    -----
            # -----  python command, check out commit hash and continue  -----
            # -----  comet exp                                           -----
            # ----------------------------------------------------------------
            sbatch_files = list(run.glob("*.sh"))
            assert len(sbatch_files) == 1, "{} .sh files in {}".format(
                len(sbatch_files), run
            )

            sbatch_file = sbatch_files[0]
            with sbatch_file.open("r") as f:
                lines = f.readlines()

            git_hash = get_text(run / "hash.txt")
            comet_id = get_text(run / "run_id.txt")

            new_lines = [
                l
                if opts.python_command not in l
                else process_to_resume(l, git_hash, comet_id)
                for l in lines
            ]
            resume_sbatch_file = sbatch_file.parent / (sbatch_file.stem + "_resume.sh")
            with (resume_sbatch_file).open("w") as f:
                f.writelines(new_lines)
            if not opts.test_mode:
                print(
                    subprocess.check_output(
                        f"sbatch {str(resume_sbatch_file)}", shell=True
                    )
                )
