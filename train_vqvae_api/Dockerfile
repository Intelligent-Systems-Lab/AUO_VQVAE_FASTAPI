# 使用帶有 Conda 的基礎映像
FROM continuumio/miniconda3

# 複製代碼和依賴文件
COPY code /code
COPY environment.yml /environment.yml

# 安裝系統依賴
RUN apt update -y && apt upgrade -y
RUN apt-get install ffmpeg libsm6 libxext6 -y

# 創建 Conda 環境並安裝依賴
RUN conda env create -f /environment.yml

# 確保 Conda 環境中的 gunicorn 在 PATH 中
RUN echo "source activate vqvae" > ~/.bashrc
ENV PATH=/opt/conda/envs/vqvae/bin:$PATH

# 設置工作目錄
WORKDIR /code
CMD ["uvicorn", "train_vqvae_api:app", "--host", "0.0.0.0", "--port", "8008", "--reload"]

