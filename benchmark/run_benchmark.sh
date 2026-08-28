#!/bin/bash

# Benchmark script for depth-detect
# Usage: ./run_benchmark.sh [duration_seconds] [output_file]
# Note: Requires sudo privileges for nvpmodel and jetson_clocks

DURATION=${1:-30}
OUTPUT_FILE=${2:-benchmark_$(date +%Y%m%d_%H%M%S).txt}
VIDEO_PATH="../data/1shu_east_0514.mp4"
CONFIG_PATH="../bin/config.yaml"
MAIN_PROGRAM="../bin/main"

echo "Starting benchmark..." | tee $OUTPUT_FILE
echo "Duration: $DURATION seconds" | tee -a $OUTPUT_FILE
echo "Video: $VIDEO_PATH" | tee -a $OUTPUT_FILE
echo "Config: $CONFIG_PATH" | tee -a $OUTPUT_FILE
echo "========================================" | tee -a $OUTPUT_FILE

# Set performance mode (requires sudo)
echo "Setting performance mode..." | tee -a $OUTPUT_FILE
sudo nvpmodel -m 0 2>&1 | tee -a $OUTPUT_FILE
sudo jetson_clocks 2>&1 | tee -a $OUTPUT_FILE

# Start tegrastats in background
echo "Starting tegrastats monitoring..." | tee -a $OUTPUT_FILE
timeout $DURATION tegrastats > tegrastats_$(date +%Y%m%d_%H%M%S).log 2>&1 &
TEGRASTATS_PID=$!

# Run the program
echo "Running depth-detect..." | tee -a $OUTPUT_FILE
echo "========================================" | tee -a $OUTPUT_FILE
timeout $DURATION $MAIN_PROGRAM $VIDEO_PATH $CONFIG_PATH 2>&1 | tee -a $OUTPUT_FILE

# Stop tegrastats
kill $TEGRASTATS_PID 2>/dev/null

echo "========================================" | tee -a $OUTPUT_FILE
echo "Benchmark completed" | tee -a $OUTPUT_FILE
echo "Results saved to: $OUTPUT_FILE" | tee -a $OUTPUT_FILE
echo "Tegrastats log: tegrastats_*.log" | tee -a $OUTPUT_FILE
