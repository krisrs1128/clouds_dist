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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--exp_dir",
        type=str,
        help="where to find the exp_dir where the experiments are stored",
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
        # -----------------------------------------------------------------
        # -----   If there's an existing resume sbatch file, use it   -----
        # -----------------------------------------------------------------
        resume_sbatch_files = list(run.glob("*_resume.sh"))
        assert (
            len(resume_sbatch_files) <= 1
        ), "More than 1 _resume.sh line in {}".format(run)

        if len(resume_sbatch_files) == 1:
            resume_sbatch_file = resume_sbatch_files[0]
            print(
                subprocess.check_output(f"sbatch {str(resume_sbatch_file)}", shell=True)
            )
        else:
            # ----------------------------------------------------------------
            # -----   Otherwise create one by adding "--resume" to the   -----
            # -----   python command                                     -----
            # ----------------------------------------------------------------
            sbatch_files = list(run.glob("*.sh"))
            assert len(sbatch_files) == 1, "{} .sh files in {}".format(
                len(sbatch_files), run
            )

            sbatch_file = sbatch_files[0]
            with sbatch_file.open("r") as f:
                lines = f.readlines()

            new_lines = [
                l
                if opts.python_command not in l
                else l[:-1].replace("\\", "") + " --resume \\\n"
                for l in lines
            ]
            resume_sbatch_file = sbatch_file.parent / (sbatch_file.stem + "_resume.sh")
            with (resume_sbatch_file).open("w") as f:
                f.write(new_lines)

            print(
                subprocess.check_output(f"sbatch {str(resume_sbatch_file)}", shell=True)
            )
