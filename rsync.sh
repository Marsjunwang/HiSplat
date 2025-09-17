# A800
rsync -avz --progress --exclude='outputs' \
    --exclude='*.pyc' --exclude='__pycache__' \
    --exclude='.git' --exclude='latest-run'\
    -e "ssh -p 30343" ./ root@10.8.150.32:/mnt/data/jun.wang03/HiSplat/

# # L40
# rsync -avz --progress --exclude='outputs' \
#     --exclude='*.pyc' --exclude='__pycache__' --exclude='build' \
#     -e "ssh" ./ narwal@192.168.31.214:/home/narwal/HiSplat/
