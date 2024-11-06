## 從github clone下，到cd到train_vqvae_api路徑  
```
cd /AUO_VQVAE_FASTAPI/inference_vqvae_api
```

## 建立image，在terminal輸入
```
docker build -t vqvae_inference_image .
```

## 建立container，在terminal輸入
```
docker run --gpus all -it -p 8008:8008  vqvae_inference_image
```

## 在網址欄中輸入就可以進到api網頁
host_url:8008/docs  
e.g. http://hc8.isl.lab.nycu.edu.tw:8008/docs
