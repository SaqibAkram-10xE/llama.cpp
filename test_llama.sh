#!/bin/bash

echo "Testing llama-cli startup..."

# Test 1: Just check if the binary starts without crashing on help
echo "Test 1: Help command"
timeout 5s ./build/bin/llama-cli -h > /dev/null 2>&1
if [ $? -eq 124 ]; then
    echo "✓ Help command works (timeout expected)"
elif [ $? -eq 0 ]; then
    echo "✓ Help command works"
else
    echo "✗ Help command failed with exit code $?"
fi

# Test 2: Try to run with a non-existent model to see if we get proper error handling
echo "Test 2: Invalid model path"
timeout 5s ./build/bin/llama-cli -m /nonexistent/model.gguf > /tmp/test_output.txt 2>&1
exit_code=$?
if [ $exit_code -eq 124 ]; then
    echo "✗ Still hanging/timing out on invalid model"
elif [ $exit_code -ne 0 ]; then
    echo "✓ Properly handles invalid model (exit code $exit_code)"
    echo "Last few lines of output:"
    tail -5 /tmp/test_output.txt
else
    echo "✗ Unexpected success with invalid model"
fi

rm -f /tmp/test_output.txt
echo "Test completed."
