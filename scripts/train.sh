#!/bin/bash

echo "Train - Random Deployment"
python3 ./random_deployment.py --json-file "train_K5_N20_M1_shadow10_episode4-5000_travelIND_fd10" --num-sim 0 &
python3 ./random_deployment.py --json-file "train_K5_N20_M2_shadow10_episode4-5000_travelIND_fd10" --num-sim 0 &
python3 ./random_deployment.py --json-file "train_K5_N20_M4_shadow10_episode4-5000_travelIND_fd10" --num-sim 0 &
python3 ./random_deployment.py --json-file "train_K10_N50_M1_shadow10_episode4-5000_travelIND_fd10" --num-sim 0 &
python3 ./random_deployment.py --json-file "train_K10_N50_M2_shadow10_episode4-5000_travelIND_fd10" --num-sim 0 &
python3 ./random_deployment.py --json-file "train_K10_N50_M4_shadow10_episode4-5000_travelIND_fd10" --num-sim 0 &
python3 ./random_deployment.py --json-file "train_K10_N50_M5_shadow10_episode4-5000_travelIND_fd10" --num-sim 0 &
python3 ./random_deployment.py --json-file "train_K10_N50_M10_shadow10_episode4-5000_travelIND_fd10" --num-sim 0 &
wait
 
echo "Train - Proposed Algorithm"
python3 ./trainProposed.py --json-file "train_K5_N20_M1_shadow10_episode4-5000_travelIND_fd10" --num-sim 0 &
python3 ./trainProposed.py --json-file "train_K5_N20_M2_shadow10_episode4-5000_travelIND_fd10" --num-sim 0 &
python3 ./trainProposed.py --json-file "train_K5_N20_M4_shadow10_episode4-5000_travelIND_fd10" --num-sim 0 &
python3 ./trainProposed.py --json-file "train_K10_N50_M1_shadow10_episode4-5000_travelIND_fd10" --num-sim 0 &
python3 ./trainProposed.py --json-file "train_K10_N50_M2_shadow10_episode4-5000_travelIND_fd10" --num-sim 0 &
python3 ./trainProposed.py --json-file "train_K10_N50_M4_shadow10_episode4-5000_travelIND_fd10" --num-sim 0 &
python3 ./trainProposed.py --json-file "train_K10_N50_M5_shadow10_episode4-5000_travelIND_fd10" --num-sim 0 &
python3 ./trainProposed.py --json-file "train_K10_N50_M10_shadow10_episode4-5000_travelIND_fd10" --num-sim 0 &

echo "Train - Joint Algorithm"
python3 ./trainJoint.py --json-file "train_K5_N20_M1_shadow10_episode4-5000_travelIND_fd10" --num-sim 0 &
python3 ./trainJoint.py --json-file "train_K5_N20_M2_shadow10_episode4-5000_travelIND_fd10" --num-sim 0 &
python3 ./trainJoint.py --json-file "train_K5_N20_M4_shadow10_episode4-5000_travelIND_fd10" --num-sim 0 &
python3 ./trainJoint.py --json-file "train_K10_N50_M1_shadow10_episode4-5000_travelIND_fd10" --num-sim 0 &
python3 ./trainJoint.py --json-file "train_K10_N50_M2_shadow10_episode4-5000_travelIND_fd10" --num-sim 0 &
python3 ./trainJoint.py --json-file "train_K10_N50_M4_shadow10_episode4-5000_travelIND_fd10" --num-sim 0 &
python3 ./trainJoint.py --json-file "train_K10_N50_M5_shadow10_episode4-5000_travelIND_fd10" --num-sim 0 &
python3 ./trainJoint.py --json-file "train_K10_N50_M10_shadow10_episode4-5000_travelIND_fd10" --num-sim 0 &
wait

echo "Train - Get FP Benchmarks"
python3 ./get_benchmarks.py --json-file "train_K5_N20_M1_shadow10_episode4-5000_travelIND_fd10" --num-sim 0 &
python3 ./get_benchmarks.py --json-file "train_K5_N20_M2_shadow10_episode4-5000_travelIND_fd10" --num-sim 0 &
python3 ./get_benchmarks.py --json-file "train_K5_N20_M4_shadow10_episode4-5000_travelIND_fd10" --num-sim 0 &
python3 ./get_benchmarks.py --json-file "train_K10_N50_M1_shadow10_episode4-5000_travelIND_fd10" --num-sim 0 &
python3 ./get_benchmarks.py --json-file "train_K10_N50_M2_shadow10_episode4-5000_travelIND_fd10" --num-sim 0 &
python3 ./get_benchmarks.py --json-file "train_K10_N50_M4_shadow10_episode4-5000_travelIND_fd10" --num-sim 0 &
python3 ./get_benchmarks.py --json-file "train_K10_N50_M5_shadow10_episode4-5000_travelIND_fd10" --num-sim 0 &
python3 ./get_benchmarks.py --json-file "train_K10_N50_M10_shadow10_episode4-5000_travelIND_fd10" --num-sim 0 &
wait

 
echo "Train Results, each interactive Ctrl +Z to pass to the next result"
python3 -i ./train_results.py --json-file "train_K5_N20_M1_shadow10_episode4-5000_travelIND_fd10"
python3 -i ./train_results.py --json-file "train_K5_N20_M2_shadow10_episode4-5000_travelIND_fd10" 
python3 -i ./train_results.py --json-file "train_K5_N20_M4_shadow10_episode4-5000_travelIND_fd10" 
python3 -i ./train_results.py --json-file "train_K10_N50_M1_shadow10_episode4-5000_travelIND_fd10"
python3 -i ./train_results.py --json-file "train_K10_N50_M2_shadow10_episode4-5000_travelIND_fd10"
python3 -i ./train_results.py --json-file "train_K10_N50_M4_shadow10_episode4-5000_travelIND_fd10" 
python3 -i ./train_results.py --json-file "train_K10_N50_M5_shadow10_episode4-5000_travelIND_fd10"
python3 -i ./train_results.py --json-file "train_K10_N50_M10_shadow10_episode4-5000_travelIND_fd10"

