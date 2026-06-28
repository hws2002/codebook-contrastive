cat > /data/guozhiqiang/hanyoushuo/multimodal/KnowCoL/.gitignore << 'EOF'
  dataset/
  dataset_01/
  dataset_mini/
  dataset_shard0104/
  checkpoints/
  knowcol.egg-info/
  __pycache__/
  *.pyc
  *.pth
  *.ckpt
  *.npy
  wandb/
  .cache/
  figures/
  scripts/
EOF

  alias gitkc='GIT_CONFIG_GLOBAL=/tmp/gitcfg_kc git -C /data/guozhiqiang/hanyoushuo/multimodal/KnowCoL'

  gitkc remote add myfork https://github.com/hws2002/KnowCoL.git
  gitkc add -A
  gitkc commit -m "training fixes: blank image fallback, configs, wandb entity hanys21"
  gitkc push myfork main

  그 다음 codebook-contrastive도 push:

  gitcb add scripts/train.py download-scripts/01_clone.sh
  gitcb commit --amend --no-edit
  gitcb push --force