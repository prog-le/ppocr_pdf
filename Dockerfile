# =============================================================================
# 基础镜像参数
#   CPU 默认： python:3.11-slim（轻量，不含 CUDA/cuDNN，120MB）
#   GPU 推荐： nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04（含 CUDA 11.8 + cuDNN 8.9）
#   构建示例： docker build --build-arg BASE_IMAGE=nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 \
#                          --build-arg PADDLE_PACKAGE=paddlepaddle-gpu \
#                          -t paddleocr-pdf:gpu .
# =============================================================================
ARG BASE_IMAGE=python:3.11-slim
FROM $BASE_IMAGE

# 设置环境变量
#   FLAGS_enable_pir_api=0 — 防止PIR执行器崩溃（必须早于任何paddle import）
#   PADDLEX_HOME           — 模型缓存目录，与三个脚本中 setdefault 一致
ENV PYTHONUNBUFFERED=1 \
    DISABLE_MODEL_SOURCE_CHECK=True \
    FLAGS_enable_pir_api=0 \
    PADDLEX_HOME=/app/.paddlex \
    PORT=8000 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC

# 构建参数
#   PADDLE_PACKAGE — CPU 版 paddlepaddle / GPU 版 paddlepaddle-gpu
#   PIP_INDEX_URL / PIP_TRUSTED_HOST — pip 镜像源（中国用户可传阿里云/清华等）
ARG PADDLE_PACKAGE=paddlepaddle
ARG PIP_INDEX_URL=https://pypi.org/simple/
ARG PIP_TRUSTED_HOST=pypi.org

# ── Python 3.11 安装（CUDA 基础镜像无 Python，需从 deadsnakes 安装）──────
RUN if ! command -v python3 &> /dev/null; then \
        apt-get update && \
        apt-get install -y --no-install-recommends software-properties-common curl ca-certificates && \
        add-apt-repository -y ppa:deadsnakes/ppa && \
        apt-get update && \
        apt-get install -y --no-install-recommends python3.11 python3.11-distutils && \
        curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 && \
        update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
        update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
        rm -rf /var/lib/apt/lists/*; \
    fi

# 设置 pip 镜像（在安装 Python 依赖前生效）
# 使用 python3 -m pip 确保在两种基础镜像上都能工作
RUN python3 -m pip config set global.index-url ${PIP_INDEX_URL} \
    && python3 -m pip config set global.trusted-host ${PIP_TRUSTED_HOST}

# 安装系统运行时依赖（仅运行时必需，无需 gcc/g++ 等编译工具）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    tini \
    && rm -rf /var/lib/apt/lists/*

# 创建非root用户及其同名组（CUDA 镜像无此用户）
RUN useradd -m -u 1000 -U paddleocr 2>/dev/null; id paddleocr

# 设置工作目录
WORKDIR /app

# 先复制 requirements.txt，利用 Docker 缓存层避免代码变更后重装全部依赖
COPY requirements.txt ./

# 安装 Python 依赖
RUN python3 -m pip install --no-cache-dir ${PADDLE_PACKAGE} && \
    python3 -m pip install --no-cache-dir -r requirements.txt

# 复制项目代码（含 device_utils.py，这是所有脚本的公共依赖）
COPY ocr_pdf.py api.py download_models.py device_utils.py ./

# 创建数据目录并修正权限
RUN mkdir -p /app/.paddlex /app/output /app/logs \
    && chown -R paddleocr /app

# 切换到非root用户
USER paddleocr

# 暴露端口
EXPOSE $PORT

# 使用 tini 作为入口点（确保信号正确处理和僵尸进程回收）
ENTRYPOINT ["tini", "--"]

# 默认启动 API 服务；可通过 CMD 覆盖来运行其他脚本
CMD ["python3", "api.py"]
