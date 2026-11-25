# Session setup for bash/zsh:  source env.sh   (works from any directory)
_here="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
source "$_here/.venv/bin/activate"
export PYTHONPATH="$_here/src"
export WANDB_MODE=offline
unset _here
