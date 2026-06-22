# 使用基于Debian的Python镜像，更好兼容机器学习库
FROM python:3.11-slim

# 设置环境变量
#   FLAGS_enable_pir_api=0 — 防止PIR执行器崩溃（必须早于任何paddle import）
#   PADDLEX_HOME           — 模型缓存目录，与三个脚本中 setdefault 一致
ENV PYTHONUNBUFFERED=1 \
    DISABLE_MODEL_SOURCE_CHECK=True \
    FLAGS_enable_pir_api=0 \
    PADDLEX_HOME=/app/.paddlex \
    PORT=8000

# 构建参数：构建时可指定 GPU 版本
#   docker build --build-arg PADDLE_PACKAGE=paddlepaddle-gpu -t paddleocr-pdf:gpu .
# 注意：
#   - paddlepaddle-gpu 通过 pip 自带 CUDA 12 运行时，python:3.11-slim 即可支持 GPU
#   - 运行时必须加 --gpus all --shm-size=8g，否则自动降级为 CPU
#   - 默认值 paddlepaddle 为 CPU 版本，体积更小、兼容性更广
ARG PADDLE_PACKAGE=paddlepaddle
# pip 镜像源构建参数（中国用户可传入阿里云/清华/ModelScope 等镜像）
ARG PIP_INDEX_URL=https://pypi.org/simple/
ARG PIP_TRUSTED_HOST=pypi.org
# 设置 pip 镜像（在安装 Python 依赖前生效）
RUN pip config set global.index-url ${PIP_INDEX_URL} \
    && pip config set global.trusted-host ${PIP_TRUSTED_HOST}

# 安装系统运行时依赖（仅运行时必需，无需 gcc/g++ 等编译工具）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    tini \
    && rm -rf /var/lib/apt/lists/*

# 创建非root用户
RUN useradd -m -u 1000 paddleocr

# 设置工作目录
WORKDIR /app

# 先复制 requirements.txt，利用 Docker 缓存层避免代码变更后重装全部依赖
COPY requirements.txt ./

# 安装 Python 依赖
RUN pip install --no-cache-dir ${PADDLE_PACKAGE} && \
    pip install --no-cache-dir -r requirements.txt

# 复制项目代码（含 device_utils.py，这是所有脚本的公共依赖）
COPY ocr_pdf.py api.py download_models.py device_utils.py ./

# 创建数据目录并修正权限
RUN mkdir -p /app/.paddlex /app/output /app/logs \
    && chown -R paddleocr:paddleocr /app

# 切换到非root用户
USER paddleocr

# 暴露端口
EXPOSE $PORT

# 使用 tini 作为入口点（确保信号正确处理和僵尸进程回收）
ENTRYPOINT ["tini", "--"]

# 默认启动 API 服务；可通过 CMD 覆盖来运行其他脚本
CMD ["python", "api.py"]
