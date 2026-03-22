#!/bin/bash
# Create a new experiment directory for an agent run.
#
# Experiments are created OUTSIDE the git repo to ensure:
# 1. No shared auto-memory (agents can't see prior results)
# 2. No access to other experiments or repo files
# 3. Complete isolation — agent only sees its own directory
#
# Usage:
#   ./create_experiment.sh <task> <agent-tag>
#
# Examples:
#   ./create_experiment.sh gamma haiku-20mar
#   ./create_experiment.sh composition opus-21mar
#
# Creates: ../astro-bench-experiments/<task>-<agent-tag>/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXPERIMENTS_ROOT="$(dirname "$SCRIPT_DIR")/astro-bench-experiments"

if [ $# -ne 2 ]; then
    echo "Usage: $0 <task> <agent-tag>"
    echo "  task: gamma | composition"
    echo "  agent-tag: e.g. haiku-20mar, sonnet-21mar, opus-22mar"
    exit 1
fi

TASK="$1"
TAG="$2"
EXPERIMENT_DIR="$EXPERIMENTS_ROOT/${TASK}-${TAG}"
TASK_DIR="$SCRIPT_DIR/$TASK"
SHARED_DATA="$SCRIPT_DIR/$TASK/data"

# Validate task
if [ ! -d "$TASK_DIR" ]; then
    echo "Error: task directory $TASK_DIR does not exist"
    echo "Available tasks: gamma, composition"
    exit 1
fi

# Check if experiment already exists
if [ -d "$EXPERIMENT_DIR" ]; then
    echo "Error: $EXPERIMENT_DIR already exists"
    exit 1
fi

# Ensure shared data exists
if [ ! -d "$SHARED_DATA" ]; then
    echo "Shared data not found. Running download_data.py..."
    (cd "$TASK_DIR" && uv run python download_data.py)
fi

if [ ! -d "$SHARED_DATA" ]; then
    echo "Error: download_data.py did not create $SHARED_DATA"
    exit 1
fi

# Create experiment directory (outside git repo)
mkdir -p "$EXPERIMENTS_ROOT"
echo "Creating experiment: $EXPERIMENT_DIR"
mkdir -p "$EXPERIMENT_DIR"

# Copy code files
cp "$TASK_DIR/CLAUDE.md" "$EXPERIMENT_DIR/"
cp "$TASK_DIR/verify.py" "$EXPERIMENT_DIR/"
cp "$TASK_DIR/download_data.py" "$EXPERIMENT_DIR/"
cp "$TASK_DIR/load_data.py" "$EXPERIMENT_DIR/"
# Baseline script at root level
cp "$TASK_DIR/baseline/"*.py "$EXPERIMENT_DIR/" 2>/dev/null || true

# Create AGENTS.md symlink
ln -sf CLAUDE.md "$EXPERIMENT_DIR/AGENTS.md"

# Copy project config for uv
cp "$SCRIPT_DIR/pyproject.toml" "$EXPERIMENT_DIR/"
cp "$SCRIPT_DIR/uv.lock" "$EXPERIMENT_DIR/"

# Symlink data to shared location
ln -sf "$SHARED_DATA" "$EXPERIMENT_DIR/data"

# Initialize uv environment
echo "Setting up Python environment..."
(cd "$EXPERIMENT_DIR" && uv sync --quiet 2>/dev/null || true)

echo ""
echo "Created: $EXPERIMENT_DIR"
echo "  Code:     copied (CLAUDE.md, verify.py, load_data.py, train_baseline.py)"
echo "  Data:     symlinked → $SHARED_DATA"
echo "  Python:   uv environment ready"
echo "  Isolated: outside git repo (no shared memory)"
echo ""
echo "To launch an agent:"
echo "  cd $EXPERIMENT_DIR"
echo "  claude"
echo "  Prompt: see $SCRIPT_DIR/prompts.md"
