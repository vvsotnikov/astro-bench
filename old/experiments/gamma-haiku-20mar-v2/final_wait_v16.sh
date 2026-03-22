#!/bin/bash
echo "=== FINAL WAIT FOR V16 COMPLETION ==="
echo "v16 (seed=456) has been training for ~15 minutes"
echo "Expected completion in ~10 minutes..."
echo ""

# Wait with a timeout of 20 minutes
max_wait=1200
elapsed=0
interval=30

while [ $elapsed -lt $max_wait ]; do
  if [ -f predictions_v16.npz ]; then
    echo "✓✓✓ V16 COMPLETE! ✓✓✓"
    echo ""
    tail -1 results.tsv | awk -F'\t' '{
      print "Attempt:", $1
      print "Metric: " $2
      print "Description: " $4
    }'
    exit 0
  fi
  
  remaining=$((max_wait - elapsed))
  echo "[$(date '+%H:%M:%S')] Still training... $(($remaining / 60)) min remaining"
  sleep $interval
  elapsed=$((elapsed + interval))
done

echo "Timeout: v16 did not complete within 20 minutes"
exit 1
