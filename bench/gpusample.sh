#!/bin/bash
# 0.5 s samples: epoch, gpu util %, mem-ctrl util %, sm MHz, W, then CPU ticks of the given pids and RDMA bytes.
# Usage: gpusample.sh <outfile> <pid>[,<pid>...]
OUT=$1; PIDS=${2:-}
HCA=/sys/class/infiniband/rocep1s0f1/ports/1/counters
while true; do
  g=$(nvidia-smi --query-gpu=utilization.gpu,utilization.memory,clocks.sm,power.draw --format=csv,noheader,nounits | tr -d ' ')
  c=""
  for p in ${PIDS//,/ }; do
    if [ -r /proc/$p/stat ]; then c="$c,$(awk '{print $14+$15}' /proc/$p/stat)"; else c="$c,NA"; fi
  done
  r=""
  if [ -r $HCA/port_xmit_data ]; then r=",$(cat $HCA/port_xmit_data),$(cat $HCA/port_rcv_data)"; fi
  echo "$(date +%s.%N),$g$c$r" >> $OUT
  sleep 0.5
done
