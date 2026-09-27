rsync -avz --progress --partial \
  --exclude-from='upload_exclusions.txt' \
  -e ssh \
  /mnt/c/Users/nikob/Documents/Coding/masters/VividFace/ \
  dhoulety@nibi.alliancecan.ca:/home/dhoulety/projects/def-srl/VividFace/ 