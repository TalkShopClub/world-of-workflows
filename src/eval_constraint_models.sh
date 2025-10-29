#!/bin/bash

# Run the evaluation pipeline for all the models in parallel

# List of models to evaluate
models=(
    # "openai/gpt-4o"
    # "openai/gpt-4o-mini"
    "openai/o3"
    "anthropic/claude-sonnet-4"
    "openai/gpt-5"
    "anthropic/claude-sonnet-4.5"
    "google/gemini-2.5-pro"
)

# Function to run evaluation for a single model
run_model_evaluation() {
    local model=$1
    echo "Starting evaluation for model: $model"
    
    # Run the evaluation pipeline
    python eval_pipeline.py --model "$model" --perfect_schema true --mode action_only --trajectory_type combined
    
    echo "Completed evaluation for model: $model"
}

# Export the function so it can be used by parallel processes
export -f run_model_evaluation

echo "Starting parallel evaluation of ${#models[@]} models..."

# Create PID file to store process IDs
PID_FILE="eval_model_pids.txt"
echo "# Model evaluation PIDs - $(date)" > "$PID_FILE"
echo "# Use 'kill -TERM $(cat $PID_FILE | grep -v '^#' | awk '{print $2}')' to stop all processes" >> "$PID_FILE"

# Run all models in parallel using background processes
pids=()
for model in "${models[@]}"; do
    run_model_evaluation "$model" &
    pid=$!
    pids+=($pid)
    echo "Started background process for $model (PID: $pid)"
    # Store PID with model name in file
    echo "$model $pid" >> "$PID_FILE"
done

echo "All models started. Waiting for completion..."

# Wait for all background processes to complete
for pid in "${pids[@]}"; do
    wait $pid
    echo "Process $pid completed"
done

echo "All model evaluations completed!"

# Clean up PID file
rm -f "$PID_FILE"
echo "Cleaned up PID file: $PID_FILE"

# Optional: Display summary of results
echo ""
echo "=== EVALUATION SUMMARY ==="
echo "Models evaluated: ${#models[@]}"
for model in "${models[@]}"; do
    echo "  - $model"
done
echo "Check the llm_evals directory for detailed results." 