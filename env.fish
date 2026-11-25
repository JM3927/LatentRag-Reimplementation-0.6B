# Session setup for fish:  source env.fish   (works from any directory)
set -l here (realpath (dirname (status --current-filename)))
source $here/.venv/bin/activate.fish
set -gx PYTHONPATH $here/src
set -gx WANDB_MODE offline
