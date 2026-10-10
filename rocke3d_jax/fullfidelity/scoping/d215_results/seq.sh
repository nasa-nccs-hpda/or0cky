#!/bin/bash
cd /panfs/ccds02/nobackup/people/gtamkin/.nccstmp/claude-855113861/-panfs-ccds02-nobackup-people-gtamkin-dev-ilab-agentic-ai-ilab-agentic-ai-projects-imvi-rocke3d-jax/170e1eae-6e03-4e76-bfe8-7e8523bf7a9a/scratchpad/d215
for t in ens2 ens3; do (uptime; ps -eo psr,pcpu,comm --sort=-pcpu | head -4) > pre_$t.txt; ./one.sh $t /panfs/ccds02/nobackup/people/gtamkin/.nccstmp/claude-855113861/-panfs-ccds02-nobackup-people-gtamkin-dev-ilab-agentic-ai-ilab-agentic-ai-projects-imvi-rocke3d-jax/170e1eae-6e03-4e76-bfe8-7e8523bf7a9a/scratchpad/d215/../ens_jan/run_ctrl/P2SAoM40.bin 7; done
for t in pris1 pris2 pris3; do (uptime; ps -eo psr,pcpu,comm --sort=-pcpu | head -4) > pre_$t.txt; ./one.sh $t /panfs/ccds02/nobackup/people/gtamkin/dev/modelE2_planet_2.0/ModelE_Support/huge_space/P2SAoM40/P2SAoM40.bin 7; done
touch seq.done
