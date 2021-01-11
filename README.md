# var_autoencoder
Variational autoencoder with TF2

+ Clone this repo, cd into 'files' folder

+ Start Vitis-AI GPU docker
```shell
./docker_run.sh xilinx/vitis-ai-gpu:latest
```

+ Training
```shell
python training.py
```

+ Quantization
```shell
python quantize.py
```

