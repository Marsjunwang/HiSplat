# A800
rsync -avz --progress --exclude='outputs' \
    --exclude='*.pyc' --exclude='__pycache__' \
    --exclude='.git' --exclude='latest-run'\
    -e "ssh -p 30717" ./ root@10.8.150.32:/mnt/data/jun.wang03/HiSplat/

# # L40 # --exclude='*.pyc' --exclude='__pycache__' --exclude='build' \
# rsync -avz --progress --exclude='outputs' \
#     -e "ssh" ./ narwal@192.168.31.214:/home/narwal/HiSplat/
