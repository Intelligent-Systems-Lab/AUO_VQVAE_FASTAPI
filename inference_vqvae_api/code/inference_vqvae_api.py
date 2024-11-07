# FastAPI
from typing import Union
from enum import Enum
from fastapi import FastAPI, File, UploadFile, Response, status, BackgroundTasks
from fastapi.responses import FileResponse
import cv2
import numpy as np
import json
import zipfile
import os
import aiofiles
import shutil
import sys
import csv
import torch
import yaml
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import cv2
import timm
from torch.nn import DataParallel
import uvicorn
import logging
from multiprocessing import Process, Array, Lock, Manager
import io
import time
import os
import torch
import torchvision
from modules import VectorQuantizedVAE
import torch.nn.functional as F
from argparse import Namespace
import subprocess

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s,%(msecs)03d [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
    )
logger = logging.getLogger(__name__)

app = FastAPI()
lock = Lock()
download_dir = "./download"
weight_dir = "./weights"
weight_info = "./weights/info.json"
result_dir = "./result"
data_dir = './data'
image_type = ('.png', '.jpg', '.jpeg', '.bmp')
weight_type = ('h5', 'ckpt', 'pth', 'pt', 'tar')

manager = Manager()
share_models = manager.list([])
process_model = [{'model_id' : None, 'model' : None}]


class Device(str, Enum):
    cpu = "cpu"
    gpu = "gpu"

# 轉換影像的變換過程
transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def _critical_section_update_weight_list(weight_list_id, w):
    with lock:
        # print(f"Process {os.getpid()} is entering the critical section. (PID: {os.getpid()})")
        logger.info(f"Process {os.getpid()} is entering the critical section. (PID: {os.getpid()})")
        weight_list = _get_weight_list()
        weight_list[weight_list_id] = w
        _updata_weight_list(weight_list)
        logger.info(f"Process {os.getpid()} is leaving the critical section. (PID: {os.getpid()})")
        # print(f"Process {os.getpid()} is leaving the critical section. (PID: {os.getpid()})")

def _critical_section_weight_id():
    with lock:
        print(f"Process {os.getpid()} is entering the critical section. (PID: {os.getpid()})")
        w = {"weight_id": None, "name": None, "info": None, "file_name": None, "file_path" : None}
        # get the current weight list
        weight_list = _get_weight_list()
        # assign weight_id
        if len(weight_list):
            w["weight_id"] = weight_list[-1]["weight_id"] + 1
        else:
            w["weight_id"] = 0
        # add into weight list
        weight_list.append(w)
        # update the weight_list
        _updata_weight_list(weight_list)
        weight_list_id,_ = _get_weight_index(w["weight_id"])
        print("weight_id : " + str(w["weight_id"]) + " is created.")
        print(f"Process {os.getpid()} is leaving the critical section. (PID: {os.getpid()})")
    return weight_list_id, w

def _critical_section_share_models(model_info):
    with lock:
        print(f"Process {os.getpid()} is entering the critical section. (PID: {os.getpid()})")
        if len(share_models) == 0:
            model_info['model_id'] = 0
            share_models.append(model_info)
        else:
            model_info['model_id'] = int(len(share_models))
            share_models.append(model_info)
        print("model ID " + str(model_info['model_id']) + " is created.")
        print(f"Process {os.getpid()} is leaving the critical section. (PID: {os.getpid()})")
    return model_info['model_id']

def _get_weight_list():
    if not os.path.isfile(weight_info):
        return []
    else:
        with open(weight_info, mode='r') as file:
            weight_list = json.load(file)
        return weight_list

def _updata_weight_list(weight_list):
    with open(weight_info, mode='w') as file:
        json.dump(weight_list, file, ensure_ascii=False, indent=4)
    logger.info("weight list update!")
    # print("weight list update!")
 
'''def _load_model(w_dir, weight_info):
    config_file = os.path.join(w_dir, "config.yaml")
    with open(config_file, 'r') as stream:
        try:
            cfg = yaml.load(stream, Loader=yaml.CLoader)
        except yaml.YAMLError as exc:
            print(exc)
    model = _build_model()
    parallel_model = DataParallel(model)
    weight_path = os.path.join(w_dir, str(weight_info["file_name"]))
    weight = torch.load(weight_path, map_location = 'cpu')
    parallel_model.load_state_dict(weight, strict = False) 
    return parallel_model'''

    

def _check_weight_list(weight_list, weight_id):
    w_idx = next((i for i in range(len(weight_list))
                  if weight_list[i]['weight_id'] == weight_id), None)
    if w_idx is None:
        return False
    else:
        return True

def _get_weight_index(weight_id):
    weight_list = _get_weight_list()
    w_idx = next((i for i in range(len(weight_list))
                  if weight_list[i]['weight_id'] == weight_id), None)
    if w_idx is None:
        return w_idx, None
    else:
        return w_idx, weight_list[w_idx]


def _delayed_remove(path: str, delay: int = 10):
    time.sleep(delay)
    os.remove(path)
def _delayed_remove_dir(path: str, delay: int = 10):
    time.sleep(delay)
    shutil.rmtree(path)
@app.get("/weight/")
async def get_weight_list(response: Response):
    """Return weight list

    Args:
        response (Response): response

    Returns:
        list: [dict: weight_list, int: error_code]
    """
    weight_list = _get_weight_list()

    error_code = 0
    logger.info("get_weight_list!")
    return {"weight_list": weight_list, "error_code": error_code}

@app.post("/weight/{name}")
async def post_weight(response: Response, weight: UploadFile, name: str, info: Union[str, None] = None):
    """Received the zip file of the Weights, create weight_info, 
    assign weight_id, add weight_info into record(json file),
    store the weights into weight folder by the weight_id

    Args:
        response (Response): response
        name (str): the name of the weights
        info (sre): the annotation of the weights
        weight (UploadFile): the zip file of the weights

    Returns:
        list: [int: weight_id, int: error_code]
    """
    weight_list_id, w = _critical_section_weight_id()
    w["name"] = name
    w["info"] = info
    # Store the zip
    zip_path = os.path.join(weight_dir, "{}.zip".format(str(w["weight_id"])))

    async with aiofiles.open(zip_path, mode="wb") as out_file:
        content = await weight.read()
        await out_file.write(content)

    if not zipfile.is_zipfile(zip_path):
        os.remove(zip_path)
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {"error_code": 1, "error_msg": "Upload file is not a zip file."}

    # Find Zip Fild name
    zip_name = os.path.splitext(os.path.basename(zip_path))[0]
    # Extract the zip
    w_dir_path = os.path.join(weight_dir, str(w["weight_id"]))
    if not os.path.exists(w_dir_path):
        os.mkdir(w_dir_path)

    with zipfile.ZipFile(zip_path, mode='r') as zip_file:
        zip_file.extractall(w_dir_path)

    # 獲取解壓後的內容（資料夾或檔案）
    extracted_items = os.listdir(w_dir_path)

    # 假設解壓後的內容只有一個資料夾 a，要將該資料夾的內容移動到 w_dir_path
    if len(extracted_items) == 1:
        extracted_folder = os.path.join(w_dir_path, extracted_items[0])
        if os.path.isdir(extracted_folder):
            # 如果是資料夾，將該資料夾的內容移動到 w_dir_path
            for item in os.listdir(extracted_folder):
                source_path = os.path.join(extracted_folder, item)
                destination_path = os.path.join(w_dir_path, item)
                shutil.move(source_path, destination_path)
            # 刪除多餘的資料夾 a
            os.rmdir(extracted_folder)
            
    # Delete zip
    os.remove(zip_path)
    
    w["file_path"] = w_dir_path
    for root, dirs, files in os.walk(w_dir_path):
        for f in files:
            print(f)
            if f.endswith(weight_type):
                w["file_name"] = f
    # Update the weight_list
    _critical_section_update_weight_list(weight_list_id, w)
    
    error_code = 0
    logger.info("post_weight!")
    return {"weight_id": w["weight_id"], "error_code": error_code}



@app.delete("/weight/{weight_id}")
async def delete_weight(response: Response, weight_id: int):
    """Delete the weights of weight_id

    Args:
        response (Response): response
        weight_id (int): the weights of weight_id to be deleted

    Returns:
        int: error_code
    """
    error_code = 0
    weight_list = _get_weight_list()
    w_idx, _ = _get_weight_index(weight_id)
    if w_idx is None:
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {"error_code": 1, "error_msg": "Model is not exist."}
    else:
        del weight_list[w_idx]
        _updata_weight_list(weight_list)
        shutil.rmtree(os.path.join(weight_dir, str(weight_id)))
        logger.info("delete_weight!")
        

    return {"error_code": error_code}

@app.post("/weight/load/{weight_id}")
async def post_load_weight(response: Response, weight_id: int):
    """Load the weight of weight_id

    Args:
        response (Response): response
        weight_id (int): the weight of weight_id to be loaded
    Returns:
        int: error_code
    """

    weight_list = _get_weight_list()

    if _check_weight_list(weight_list, weight_id):
        id, weight_list = _get_weight_index(weight_id)
        model_info={
            "model_id": None,
            "name": str(weight_list['name']),
            "model_path": str(os.path.join(weight_list['file_path'], weight_list['file_name'])),
        }
        #model = _load_model(str(weight_list['file_path']), weight_list)
        #model_list = {"model_id" : None, "name" : str(weight_list['name']), "model" : model.to('cpu')}
        model_id = _critical_section_share_models(model_info)
    else:
        return {"error_code": 1, "error_msg": "Model is not exist."}

    return {"error_code": 0, "model_id": model_id}

@app.get("/weight/load/")
async def get_load_weight(response: Response):
    """Get the model list
    Args:
        response (Response): response
    Returns:
        loaded models: models_list, int: error_code
    """
    models_list = []
    for i in share_models:
        temp = {"model_id" : i['model_id'], "name" :str(i['name']), "model_path": str(i['model_path']) }
        models_list.append(temp) 
    return {"loaded models": models_list, "error_code": 0}

def find(weight_id):
    model_exist = next((item for item in share_models if item["model_id"] == weight_id), None)
    return model_exist

def delete(weight_id):
    model_index = next((id for id, item in enumerate(share_models) if item["model_id"] == weight_id), None)
    del share_models[model_index]

@app.delete("/weight/load/{weight_id}")
async def delete_load_weight(response: Response, model_id: int):
    """Unload the model

    Args:
        response (Response): response

    Returns:
        int: error_code
    """

    model_exist = find(model_id)
    if model_exist is not None:
        delete(model_id)
        global process_model
        process_model = share_models[:]
        print(len(process_model))
        torch.cuda.empty_cache()
        logger.info("delete_load_weight!")
        return {"error_code": 0}
    else:
        return {"error_code": 1, "error_code": "Model is not loaded in the memory."}


@app.post("/inference/")
async def post_inference(response: Response, 
    background_tasks: BackgroundTasks,
    name: str, data: UploadFile, 
    model_id: int,
    device: Device = "cpu",
    hidden_size: Union[int, None] = None,
    k: Union[int, None] = None,
    negative_sample_number: Union[int, None] = None,
    batch_size: Union[int, None] = None,
    ):
    """ Post the Image Folder dataset to start a training

    Args:
        response: Http Response
        name: the name of this training task
        model_id: the model id you want to used for inferenceing

    Returns:
        job_id: the job_id of the trianing process

    """
    with lock:
        # with open('./status.json', 'r') as f:
        #     idle = json.load(f)
        
        # if idle['idle'] == False:
        #     return {"error_code": 1, "error_msg": "Another job is training or testing."} 

        # init_status = dict()
        # init_status['status'] = "inferenceing"
        # init_status['idle'] = False
        # init_status['completed'] = False
        # with open('./status.json', 'w') as f:
        #     json.dump(init_status, f)
        share_model_index = next((id for id, item in enumerate(share_models) if item["model_id"] == model_id), None)
        model_info = share_models[share_model_index]
        print(model_info)
        # if not os.path.exists(weight_dir):
        #     os.mkdir(weight_dir)
        # job = {"job_id": None, "pid": None, "name": name, "info": info, "type": "train", "status": None}
            
        if device == 'cpu':
            device = "cpu"
        else:
            if torch.cuda.is_available():
                device = "cuda"

        # Download the dataset
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)

        data_zip_path = os.path.join(data_dir, str(data.filename))
        async with aiofiles.open(data_zip_path, mode="wb") as out_file:
            content = await data.read()
            await out_file.write(content)
        
        # Check if it is a zip file
        if not zipfile.is_zipfile(data_zip_path):
            os.remove(data_zip_path)
            response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
            return {"error_code": 3, "error_msg": "Upload file is not a zip file."}
        
        # Extract files
        with zipfile.ZipFile(data_zip_path, mode='r') as zip_file:
            zip_file.extractall(data_dir)
        os.remove(data_zip_path)
        data_path, _ = os.path.splitext(data_zip_path)

        print("step 2. Dataset managemet passed")
        
        with open('./configs/config.yaml', 'r') as f:
                config = yaml.load(f, Loader = yaml.FullLoader)

        config['exp_name'] = name if name is not None else config['exp_name']
        config['data_dir'] = data_path
        config['batch_size'] = batch_size if batch_size is not None else config['batch_size']
        config['hidden_size'] = hidden_size if hidden_size is not None else config['hidden_size']
        config['device'] = device if device is not None else config['device']
        config['k'] = k if k is not None else config['k']
        config['negative_sample_number'] = negative_sample_number if negative_sample_number is not None else config['negative_sample_number']
        config['model_path'] = model_info["model_path"]
        
        with open('./configs/config.yaml', 'w') as f:
            yaml.dump(config, f, sort_keys=False)

        print("step 3. Config management passed")

        # Call the watch dog program
        proc = subprocess.Popen(["python", "watchdog_2.py"], shell=False, preexec_fn=os.setsid)
        print("step 4. call watch_dog.py")
        
        # 使用 communicate() 確保進程完成執行
        stdout, stderr = proc.communicate()

        # 檢查進程的返回碼是否為 0，表示成功執行
        if proc.returncode != 0:
            print(f"watchdog_2.py 執行失敗: {stderr}")
            response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
            return {"error_code": 5, "error_msg": "Error occurred during execution of watchdog_2.py."}

        # 定義重試機制，等待壓縮檔案完成
        zip_file_path = f"./dataset/{name}.zip"
        max_retries = 5  # 重試次數
        wait_time = 5    # 每次重試等待的秒數
        img_folder = os.path.join('./dataset', name)
        time.sleep(wait_time)
        background_tasks.add_task(_delayed_remove_dir, img_folder, delay=10)
        background_tasks.add_task(_delayed_remove_dir, data_path, delay=10)
        background_tasks.add_task(_delayed_remove, zip_file_path, delay=10)
        
        with open('./status.json', 'r') as f:
            idle = json.load(f)
        for _ in range(max_retries):
            #重新讀取狀態
            with open('./status.json', 'r') as f:
                idle = json.load(f)
            if idle['completed'] == True:
                return FileResponse(zip_file_path, media_type='application/zip', filename=f"{name}.zip")
            else:
                # 如果檔案還沒生成，等待並重試
                time.sleep(wait_time)

        # 若嘗試數次後檔案依然不存在，返回錯誤訊息
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {"error_code": 4, "error_msg": "Zip file not found after waiting."}


if __name__ == "__main__":
    logger.info("Fast API Activate !!!")
