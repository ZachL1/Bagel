Build docker image:
```
docker build -t bagel:latest .
```

Run docker image at the first time:
```
docker run -it --name bg25 -p 7860:7860 -p 2222:22 -v ~/BAGEL:/workspace/aesBAGEL --gpus all --shm-size 128G bagel 
```

After first time:
```
docker start -a bg25
```

Stop the container:
```
docker stop bg25
```