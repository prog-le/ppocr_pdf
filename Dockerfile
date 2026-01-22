# 使用基于Debian的Python镜像，更好兼容机器学习库
FROM python:3.11-slim

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    DISABLE_MODEL_SOURCE_CHECK=True \
    PADDLE_HOME=/app/.paddlex \
    PORT=8000

# 安装系统依赖（简化版本，使用通用包名）
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libglib2.0-0 \
    libgl1 \
    tini \
    && rm -rf /var/lib/apt/lists/*

# 创建非root用户
RUN useradd -m -u 1000 paddleocr

# 设置工作目录
WORKDIR /app

# 复制必要的文件
COPY requirements.txt ocr_pdf.py api.py download_models.py ./

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 创建必要的目录并设置权限
RUN mkdir -p /app/.paddlex /app/output /app/logs \
    && chown -R paddleocr:paddleocr /app

# 切换到非root用户
USER paddleocr

# 暴露端口
EXPOSE $PORT

# 使用tini作为入口点
ENTRYPOINT ["tini", "--"]

# 启动命令
CMD ["python", "api.py"]