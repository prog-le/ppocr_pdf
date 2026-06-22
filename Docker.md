# PaddleOCR PDF文字识别项目 - Docker部署说明

## 1. 项目概述

这是一个基于Python 3.11和PaddleOCR的PDF文字识别项目，能够对PDF文件执行高精度文字识别操作。项目支持两种工作模式：手动模式和守护模式，满足不同场景的使用需求。

### 核心功能

- ✅ **PDF文字识别**：使用PaddleOCR库对PDF文件进行高精度文字识别，支持中英文
- ✅ **批量处理**：支持对目录中的多个PDF文件进行批量处理
- ✅ **两种工作模式**：手动模式和守护模式
- ✅ **多模型支持**：提供四种PaddleOCR模型供选择
- ✅ **PDF优化**：支持PDF文件优化，可配置不同优化级别
- ✅ **灰度渲染**：支持灰度渲染选项，减少内存占用
- ✅ **API服务**：基于FastAPI的RESTful API

## 2. Docker镜像特点

- **轻量级**：基于`python:3.11-slim`镜像，体积优化合理
- **跨平台**：支持linux/amd64（x86-64）和linux/arm64架构
- **安全可靠**：
  - 非root用户（paddleocr）运行容器
  - 使用tini作为入口点，确保容器优雅退出
  - 严格的依赖管理
- **预配置**：
  - 包含所有必要的系统和Python依赖
  - 预下载默认模型
  - 配置了合理的环境变量
- **易用性**：
  - 一键构建和运行
  - 支持持久化卷挂载
  - 灵活的环境变量配置

## 3. 环境要求

- Docker 20.10+
- 至少2GB可用内存
- 至少5GB可用磁盘空间

## 4. 部署步骤

### 4.1 克隆项目

```bash
git clone https://github.com/prog-le/ppocr_pdf.git
cd ppocr_pdf
```

### 4.2 构建Docker镜像

#### 4.2.1 构建默认架构镜像

```bash
docker build -t paddleocr-pdf .
```

#### 4.2.2 构建特定架构镜像

```bash
# 构建x86-64架构镜像
docker buildx build --platform linux/amd64 -t paddleocr-pdf:amd64 --load .

# 构建ARM64架构镜像
docker buildx build --platform linux/arm64 -t paddleocr-pdf:arm64 --load .
```


### 4.3 运行Docker容器

#### 4.3.1 基本运行

```bash
docker run -d -p 8000:8000 --name paddleocr-pdf-container paddleocr-pdf
```

#### 4.3.2 挂载持久化卷

```bash
docker run -d -p 8000:8000 \
  -v ./models:/app/.paddlex \
  -v ./output:/app/output \
  -v ./logs:/app/logs \
  --name paddleocr-pdf-container \
  paddleocr-pdf
```

**卷说明**：
- `./models:/app/.paddlex`：模型缓存目录，用于持久化保存下载的模型
- `./output:/app/output`：输出目录，用于保存OCR识别结果
- `./logs:/app/logs`：日志目录，用于保存运行日志

#### 4.3.3 配置环境变量

```bash
docker run -d -p 8000:8000 \
  -e LOG_LEVEL=info \
  -e PORT=8000 \
  -e DISABLE_MODEL_SOURCE_CHECK=True \
  --name paddleocr-pdf-container \
  paddleocr-pdf
```

**环境变量说明**：
- `LOG_LEVEL`：日志级别，可选值：debug, info, warning, error, critical
- `PORT`：API服务端口，默认：8000
- `DISABLE_MODEL_SOURCE_CHECK`：是否禁用模型源检查，默认：True
- `PYTHONUNBUFFERED`：是否启用Python无缓冲输出，默认：1

## 5. 访问服务

### 5.1 API服务

- **健康检查**：`http://localhost:8000/health`
- **API文档**：
  - Swagger UI：`http://localhost:8000/docs`
  - ReDoc：`http://localhost:8000/redoc`

### 5.2 测试API

```bash
# 健康检查
curl http://localhost:8000/health

# PDF OCR识别（基本请求）
curl -X POST "http://localhost:8000/ocr/pdf" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@./test_input/01.pdf" \
  -F "model=pp-ocrv5"

# PDF OCR识别（带优化参数）
curl -X POST "http://localhost:8000/ocr/pdf" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@./test_input/01.pdf" \
  -F "model=pp-ocrv5" \
  -F "optimize_pdf=true" \
  -F "optimize_level=high" \
  -F "grayscale=true"

# PDF OCR识别（仅输出 JSON 格式）
curl -X POST "http://localhost:8000/ocr/pdf" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@./test_input/01.pdf" \
  -F "output_formats=json"
```

## 6. 容器管理

### 6.1 查看容器状态

```bash
# 查看所有容器
 docker ps -a

# 查看容器日志
docker logs paddleocr-pdf-container

# 查看容器详情
docker inspect paddleocr-pdf-container
```

### 6.2 停止和启动容器

```bash
# 停止容器
docker stop paddleocr-pdf-container

# 启动容器
docker start paddleocr-pdf-container

# 重启容器
docker restart paddleocr-pdf-container
```

### 6.3 进入容器

```bash
# 进入运行中的容器
docker exec -it paddleocr-pdf-container sh

# 查看容器内文件
docker exec paddleocr-pdf-container ls -la /app
```

### 6.4 删除容器

```bash
# 删除停止的容器
docker rm paddleocr-pdf-container

# 强制删除运行中的容器
docker rm -f paddleocr-pdf-container
```

## 7. 镜像管理

```bash
# 查看本地镜像
docker images

# 删除本地镜像
docker rmi paddleocr-pdf

# 清理未使用的镜像
docker image prune
```

## 8. 高级配置

### 8.1 使用Docker Compose

创建`docker-compose.yml`文件：

```yaml
version: '3.8'

services:
  paddleocr-pdf:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/.paddlex
      - ./output:/app/output
      - ./logs:/app/logs
    environment:
      - LOG_LEVEL=info
      - PORT=8000
    restart: unless-stopped
    user: "1000:1000"
```

运行：
```bash
docker-compose up -d
```

### 8.2 自定义模型下载

```bash
# 自定义下载模型
docker run -it --rm \
  -v ./models:/app/.paddlex \
  paddleocr-pdf \
  python download_models.py -m pp-ocrv5,paddleocr-vl
```

### 8.3 使用私有镜像仓库

```bash
# 登录私有仓库
docker login your-registry.com

# 构建并推送
docker buildx build --platform linux/amd64,linux/arm64 -t your-registry.com/your-username/paddleocr-pdf:latest --push .

# 拉取并运行
docker pull your-registry.com/your-username/paddleocr-pdf:latest
docker run -d -p 8000:8000 your-registry.com/your-username/paddleocr-pdf:latest
```

## 9. 常见问题

### 9.1 容器启动失败

**问题**：容器启动后立即退出
**解决方案**：
```bash
# 查看日志，分析错误原因
docker logs paddleocr-pdf-container

# 尝试交互式运行，查看具体错误
docker run -it --rm paddleocr-pdf sh
```

### 9.2 端口被占用

**问题**：`Error starting userland proxy: listen tcp4 0.0.0.0:8000: bind: address already in use`
**解决方案**：
```bash
# 查看端口占用情况
lsof -i :8000

# 杀死占用端口的进程
kill -9 <PID>

# 或使用不同端口
  docker run -d -p 8080:8000 paddleocr-pdf
```

### 9.3 内存不足

**问题**：容器因内存不足被杀死
**解决方案**：
```bash
# 增加容器内存限制
docker run -d -p 8000:8000 --memory=4g --memory-swap=4g paddleocr-pdf
```

### 9.4 模型下载失败

**问题**：容器内模型下载失败
**解决方案**：
```bash
# 启用模型源检查禁用
docker run -d -p 8000:8000 -e DISABLE_MODEL_SOURCE_CHECK=True paddleocr-pdf
```

## 10. 最佳实践

1. **使用持久化卷**：
   - 挂载模型目录，避免每次重建容器都重新下载模型
   - 挂载输出和日志目录，便于数据管理

2. **合理配置资源**：
   - 根据实际需求调整容器内存限制
   - 考虑使用GPU加速（如果可用）

3. **定期更新镜像**：
   - 及时获取最新的安全补丁和功能更新
   - 使用明确的版本标签，避免使用`latest`标签

4. **监控容器状态**：
   - 使用Docker内置监控工具或第三方监控系统
   - 定期查看容器日志，及时发现问题

5. **安全配置**：
   - 避免使用root用户运行容器
   - 限制容器的网络访问权限
   - 定期更新Docker引擎

## 11. 许可证

本项目采用[Apache License 2.0](LICENSE)许可证。

## 12. 联系方式

- 项目地址：https://github.com/prog-le/ppocr_pdf
- 电子邮件：prog.le@outlook.com

---

**感谢使用PaddleOCR PDF文字识别项目！** 🚀