#!/bin/bash

echo "Table 1 - Row 1"
echo "Test - Deployment"
python3 ./random_deployment.py --json-file "test_K5_N20_M1_shadow10_episode10-500_travel0_fd10" &
wait

echo "Test - Policy"
# Run parallel instead
python3 ./testJoint.py --json-file "test_K5_N20_M1_shadow10_episode10-500_travel0_fd10" --json-files-train "train_K5_N20_M1_shadow10_episode4-5000_travelIND_fd10" --json-file-PA-train "ddpg200_100_50" --json-file-CS-train "dqn100_50_50" & 
python3 ./testProposed.py --json-file "test_K5_N20_M1_shadow10_episode10-500_travel0_fd10" --json-files-train "train_K5_N20_M1_shadow10_episode4-5000_travelIND_fd10" --json-file-PA-train "dqn200_200_100" --json-file-CS-train "dqn200_200_100" &
wait

echo "Test - Get FP WMMSE Benchmarks"
python3 ./get_benchmarks.py --json-file "test_K5_N20_M1_shadow10_episode10-500_travel0_fd10" &
wait

echo "RESULTS:"
python3 ./test_results.py --json-file "test_K5_N20_M1_shadow10_episode10-500_travel0_fd10" --json-file-train "train_K5_N20_M1_shadow10_episode4-5000_travelIND_fd10" --json-file-PA-train "ddpg200_100_50" --json-file-CS-train "dqn100_50_50" 
python3 ./test_results.py --json-file "test_K5_N20_M1_shadow10_episode10-500_travel0_fd10" --json-file-train "train_K5_N20_M1_shadow10_episode4-5000_travelIND_fd10" --json-file-PA-train "dqn200_200_100" --json-file-CS-train "dqn200_200_100"
