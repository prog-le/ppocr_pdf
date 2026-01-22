# PaddleOCR PDF文字识别项目

## 项目概述

这是一个基于Python 3.11和PaddleOCR的PDF文字识别项目，能够对指定输入目录中的PDF文件执行文字识别操作，并将识别结果保存至指定的输出文件夹。项目支持两种工作模式：手动模式和守护模式，满足不同场景的使用需求。

## 功能特性

### 🎯 核心功能

- **PDF文字识别**：使用PaddleOCR库对PDF文件进行高精度文字识别，支持中英文
- **批量处理**：支持对目录中的多个PDF文件进行批量处理
- **两种工作模式**：
  - **手动模式**：对当前输入目录中已存在的PDF文件执行一次性识别操作
  - **守护模式**：持续监控指定输入目录，当检测到新的PDF文件被添加时自动触发识别流程
- **多模型支持**：提供四种PaddleOCR模型供选择，满足不同场景需求
- **结果保存**：将识别结果按页面组织保存为文本文件，便于后续查看和处理
- **日志记录**：自动记录识别过程的详细信息，包括日期时间、文件名、文件大小、处理耗时、处理结果等
- **PDF优化**：支持PDF文件优化，可配置不同优化级别，提高处理效率
- **灰度渲染**：支持灰度渲染选项，减少内存占用，提升处理速度

### ✨ 技术特点

- **高效识别**：基于PaddleOCR深度学习框架，识别准确率高
- **命令行操作**：支持通过命令行参数灵活配置和运行
- **自动监控**：守护模式下实时监控目录变化，自动处理新文件
- **跨平台**：支持Windows、Linux和macOS操作系统
- **易于扩展**：模块化设计，便于功能扩展和定制
- **智能缓存**：自动管理模型文件缓存，避免重复下载
- **Markdown日志**：将识别结果以表格形式记录到Markdown文件，便于统计和分析

## 安装指南

### 方式一：直接安装（推荐）

#### 环境要求

- Python 3.11+
- pip 21.0+

#### 安装步骤

1. **克隆项目**

   ```bash
   git clone https://github.com/prog-le/ppocr_pdf.git
   cd ppocr_pdf
   ```
2. **创建虚拟环境**（可选但推荐）

   ```bash
   # 使用conda创建虚拟环境
   conda create -n paddleocr python=3.11
   conda activate paddleocr

   # 或使用venv创建虚拟环境
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   # 或 venv\Scripts\activate  # Windows
   ```
3. **安装依赖**

   ```bash
   pip install -r requirements.txt
   ```
4. **安装PaddleOCR和PaddlePaddle**

   ```bash
   # 安装PaddlePaddle CPU版本
   pip install paddlepaddle

   # 安装PaddleOCR
   pip install paddleocr
   ```

5. **安装额外依赖（可选，用于PP-StructureV3）**

   ```bash
   # 安装PP-StructureV3所需的额外依赖
   pip install "paddlex[ocr]"
   ```

### 方式二：Docker部署

#### 环境要求

- Docker 20.10+

#### 部署步骤

1. **克隆项目**

   ```bash
   git clone https://github.com/prog-le/ppocr_pdf.git
   cd ppocr_pdf
   ```

2. **构建Docker镜像**

   ```bash
   # 构建默认架构镜像
   docker build -t paddleocr-pdf .

   # 构建x86-64架构镜像
   docker buildx build --platform linux/amd64 -t paddleocr-pdf:amd64 --load .

   # 构建多架构镜像并推送
   docker buildx build --platform linux/amd64,linux/arm64 -t your-registry/paddleocr-pdf:latest --push .
   ```

3. **运行Docker容器**

   ```bash
   # 基本运行
   docker run -d -p 8000:8000 --name paddleocr-pdf-container paddleocr-pdf

   # 挂载持久化卷
   docker run -d -p 8000:8000 \
     -v ./models:/app/.paddlex \
     -v ./output:/app/output \
     -v ./logs:/app/logs \
     --name paddleocr-pdf-container \
     paddleocr-pdf

   # 配置环境变量
   docker run -d -p 8000:8000 \
     -e LOG_LEVEL=info \
     --name paddleocr-pdf-container \
     paddleocr-pdf
   ```

4. **验证服务**

   ```bash
   # 检查容器状态
   docker ps

   # 查看日志
   docker logs paddleocr-pdf-container

   # 测试健康检查
   curl http://localhost:8000/health
   ```

#### Docker镜像特点

- ✅ 基于轻量级`python:3.11-slim`镜像
- ✅ 非root用户运行，提高安全性
- ✅ 支持多平台架构（linux/amd64, linux/arm64）
- ✅ 包含所有必要依赖
- ✅ 预配置环境变量
- ✅ 使用tini作为入口点，确保容器优雅退出

## 使用说明

### 模型下载脚本

项目提供了`download_models.py`脚本，用于手动下载PaddleOCR模型到指定目录。

#### 脚本功能

- 支持下载三种PaddleOCR模型：`pp-ocrv5`、`pp-structurev3`、`paddleocr-vl`
- 可指定模型保存目录，方便模型管理
- 自动创建必要的目录结构
- 提供依赖检查，明确提示缺少的依赖

#### 使用方法

```bash
# 下载所有模型到默认目录(.paddlex)
python download_models.py

# 下载指定模型到自定义目录
python download_models.py -m pp-ocrv5,paddleocr-vl -o ./models

# 下载单个模型
python download_models.py -m pp-ocrv5
```

#### 参数说明

- `-m, --models`: 要下载的模型，多个模型用逗号分隔，可选值: `pp-ocrv5, pp-structurev3, paddleocr-vl, all` (下载所有模型)，默认: `all`
- `-o, --output`: 模型保存目录，默认保存到.paddlex目录

#### 注意事项

- **pp-structurev3模型**需要安装额外依赖：`pip install "paddlex[ocr]"`
- 脚本会自动检查pp-structurev3所需的依赖，并给出明确的安装提示

### 命令行参数

```bash
python ocr_pdf.py [-h] -i INPUT -o OUTPUT [-m {manual,daemon}] [-model {paddleocr-vl,pp-ocrv5,pp-structurev3,pp-chatocrv4}] [-l {debug,info,warning,error,critical}] [--optimize-pdf] [--optimize-level {low,medium,high}] [--grayscale]
```

参数说明：

- `-i, --input`: 输入目录路径（必填）
- `-o, --output`: 输出目录路径（必填）
- `-m, --mode`: 工作模式，可选值：manual（手动模式）或 daemon（守护模式），默认：manual
- `-model, --model`: OCR模型选择，可选值：paddleocr-vl、pp-ocrv5、pp-structurev3、pp-chatocrv4，默认：pp-ocrv5
- `-l, --log-level`: 日志输出级别，可选值：debug、info、warning、error、critical，默认：info
- `--optimize-pdf`: 是否优化PDF文件，默认：False
- `--optimize-level`: PDF优化级别，可选值：low、medium、high，默认：medium
- `--grayscale`: 是否使用灰度渲染，减少内存占用，默认：False
- `-h, --help`: 显示帮助信息

### 模型选择说明

程序提供四种PaddleOCR模型供选择，每种模型有其特定的适用场景：

| 模型名称       | 模型类型     | 特点                                                                      | 适用场景                                     | 支持状态                     |
| -------------- | ------------ | ------------------------------------------------------------------------- | -------------------------------------------- | ---------------------------- |
| paddleocr-vl   | 多模态模型   | 通过0.9B超紧凑视觉语言模型增强，支持109种语言，在复杂元素识别方面表现出色 | 多语言混合文档、包含表格/公式/图表的复杂文档 | ✅ 支持                      |
| pp-ocrv5       | 全场景识别   | 单模型支持五种文字类型，精度提升13个百分点，资源消耗较低                  | 普通文档识别、日常使用                       | ✅ 支持                      |
| pp-structurev3 | 复杂文档解析 | 将复杂PDF转换为保留原始结构的Markdown和JSON文件，保持文档版式和层次结构   | 结构化文档处理、需要保留格式的文档           | ✅ 支持（需额外依赖）        |
| pp-chatocrv4   | 智能信息抽取 | 原生集成ERNIE 4.5，从海量文档中精准提取关键信息，精度提升15个百分点       | 信息抽取、文档问答、关键信息提取             | ⚠️ 需API密钥，暂不直接支持 |

**注意：**
- pp-structurev3模型需要安装额外依赖：`pip install "paddlex[ocr]"`
- pp-chatocrv4模型需要配置百度千帆API密钥，目前暂不直接支持在本程序中使用

### 示例用法

#### 1. 手动模式

对手动模式下，程序会处理输入路径中的PDF文件：

```bash
# 处理目录中的所有PDF文件（默认模型）
python ocr_pdf.py -i ./test_input -o ./test_output -m manual

# 处理单个PDF文件
python ocr_pdf.py -i ./test_input/test.pdf -o ./test_output -m manual

# 使用PaddleOCR-VL模型处理目录
python ocr_pdf.py -i ./test_input -o ./test_output -m manual -model paddleocr-vl

# 使用PaddleOCR-VL模型处理单个文件
python ocr_pdf.py -i ./test_input/test.pdf -o ./test_output -m manual -model paddleocr-vl

# 使用PP-StructureV3模型处理目录
python ocr_pdf.py -i ./test_input -o ./test_output -m manual -model pp-structurev3

# 启用PDF优化（默认级别）
python ocr_pdf.py -i ./test_input -o ./test_output -m manual --optimize-pdf

# 启用PDF优化并设置高级别
python ocr_pdf.py -i ./test_input -o ./test_output -m manual --optimize-pdf --optimize-level high

# 启用灰度渲染减少内存占用
python ocr_pdf.py -i ./test_input -o ./test_output -m manual --grayscale

# 同时启用PDF优化和灰度渲染
python ocr_pdf.py -i ./test_input -o ./test_output -m manual --optimize-pdf --grayscale

# pp-chatocrv4模型需要配置API密钥，目前暂不直接支持
```

#### 2. 守护模式

在守护模式下，程序会持续监控输入目录，当检测到新的PDF文件时自动进行处理：

```bash
# 使用默认模型(pp-ocrv5)监控目录
python ocr_pdf.py -i ./test_input -o ./test_output -m daemon

# 使用指定模型监控目录
python ocr_pdf.py -i ./test_input -o ./test_output -m daemon -model paddleocr-vl

# 在守护模式下启用PDF优化
python ocr_pdf.py -i ./test_input -o ./test_output -m daemon --optimize-pdf

# 在守护模式下启用灰度渲染
python ocr_pdf.py -i ./test_input -o ./test_output -m daemon --grayscale
```

### 输出结果

识别结果会以文本文件的形式保存到指定的输出目录中，文件名与原PDF文件相同，但扩展名为 `.txt`。识别结果按页面组织，每页内容以 `=== 第 X 页 ===`分隔。

示例输出：

```
=== 第 1 页 ===
Hello World!
This is a test PDF file for OCR.
Page 1 content.
=== 第 2 页 ===
This is page 2.
More test content here.
```

### 日志记录

程序会自动记录识别过程的详细信息，包括：

- **控制台日志**：实时输出处理过程信息
- **Markdown日志**：将识别结果以表格形式记录到项目根目录下的 `ocr_logs.md` 文件

#### Markdown日志格式

```markdown
# OCR识别日志

| 日期时间 | 文件名 | 文件大小 | 总页数 | 处理耗时 | 处理结果 | 输出路径 |
|---------|------|---------|-------|---------|---------|---------|
| 2026-01-15 10:30:25 | test.pdf | 0.00MB | 2 | 8.23秒 | 成功 | ./test_output/test.txt |
| 2026-01-15 10:30:35 | document.pdf | 1.50MB | 5 | 15.42秒 | 成功 | ./test_output/document.txt |
```

#### 日志内容说明

- **日期时间**：OCR识别完成的时间
- **文件名**：被处理的PDF文件名
- **文件大小**：PDF文件的大小
- **总页数**：PDF文件的总页数
- **处理耗时**：完成整个PDF文件识别所用的时间
- **处理结果**：识别是否成功
- **输出路径**：识别结果保存的文件路径

### API服务

本项目提供了基于FastAPI的API服务，可以通过API调用OCR功能。

#### 启动API服务

```bash
python api.py
```

API服务将在 `http://localhost:8000` 启动。

#### API文档

启动服务后，可以通过以下地址访问API文档：

- **Swagger UI**：http://localhost:8000/docs
- **ReDoc**：http://localhost:8000/redoc

#### API接口

##### 1. 健康检查

```
GET /health
```

返回服务状态信息。

##### 2. PDF OCR识别

```
POST /ocr/pdf
```

**请求参数：**

- `file`：上传的PDF文件
- `model`：OCR模型选择（可选，默认：pp-ocrv5），可选值：pp-ocrv5, pp-structurev3, paddleocr-vl

**示例请求（curl）：**

```bash
curl -X POST "http://localhost:8000/ocr/pdf" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test.pdf" \
  -F "model=pp-structurev3"
```

**示例响应：**

```json
{
  "status": "success",
  "filename": "test.pdf",
  "model": "pp-structurev3",
  "result": "=== 第 1 页 ===\nHello World!\nThis is a test PDF file for OCR.\nPage 1 content.\n=== 第 2 页 ===\nThis is page 2.\nMore test content here."
}
```

## 项目结构

```
paddleocr-pdf/
├── ocr_pdf.py          # 主程序文件
├── api.py              # API服务模块
├── download_models.py  # 模型下载脚本
├── ocr_logs.md         # OCR识别日志(自动生成)
├── test_input/         # 测试输入目录
│   └── 01.pdf          # 测试PDF文件
├── test_output/        # 测试输出目录
│   └── 01.txt          # 识别结果文件
├── logs/               # 日志文件目录(自动生成)
├── .paddlex/           # 模型缓存目录(自动生成)
├── requirements.txt    # 依赖项列表
├── Dockerfile          # Docker构建文件
├── .dockerignore       # Docker忽略配置
├── .gitignore          # Git忽略配置
└── README.md           # 项目说明文档
```

## 配置方法

### 自定义OCR参数

在 `ocr_pdf.py`文件中，可以通过修改 `PDFOCRHandler`类来配置OCR识别选项：

```python
class PDFOCRHandler:
    def __init__(self, output_dir, model='pp-ocrv5'):
        self.output_dir = output_dir
        self.model = model
        
        # 根据选择的模型配置不同参数
        if model == 'paddleocr-vl':
            self.ocr = PaddleOCR(
                use_textline_orientation=True,
                lang='ch',
                text_detection_model_name='PP-OCRv5_server_det',
                text_recognition_model_name='PP-OCRv5_server_rec'
            )
        elif model == 'pp-ocrv5':
            self.ocr = PaddleOCR(
                use_textline_orientation=True,
                lang='ch'
            )
        # 其他模型配置...
  
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
```

### 配置图像缩放

为了处理超大尺寸图像导致的程序无响应问题，程序会自动检查图像尺寸并进行缩放：

```python
# 在process_pdf_page函数中
max_size = 8000  # 设置最大尺寸
height, width = img_cv.shape[:2]
if height > max_size or width > max_size:
    scale_factor = max_size / max(height, width)
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    logger.info(f"图像尺寸过大 ({width}x{height})，将缩放到 {new_width}x{new_height}")
    img_cv = cv2.resize(img_cv, (new_width, new_height), interpolation=cv2.INTER_AREA)
```

### 配置监控间隔

在守护模式下，可以通过修改 `run_daemon_mode`函数中的 `sleep`时间来调整监控间隔：

```python
def run_daemon_mode(input_dir, output_dir, model='pp-ocrv5'):
    # ...
    try:
        while True:
            time.sleep(1)  # 调整监控间隔时间（秒）
    except KeyboardInterrupt:
        observer.stop()
        logger.info("守护模式停止")
    # ...
```

## 常见问题

### 1. 权限错误

如果遇到类似 `PermissionError: [Errno 1] Operation not permitted`的错误，这通常是由于PaddleOCR尝试访问系统目录导致的。程序已经通过自定义缓存目录的方式解决了这个问题，所有缓存文件都会保存在项目目录下的 `.paddlex`文件夹中。

### 2. 识别速度慢

OCR识别速度受多种因素影响，包括PDF文件大小、页数、图像质量、CPU/GPU性能等。可以尝试以下优化方法：

- 降低PDF渲染分辨率（修改 `scale`参数）
- 使用GPU版本的PaddlePaddle
- 选择资源消耗较低的模型（如pp-ocrv5）

### 3. 识别准确率低

如果识别准确率不高，可以尝试以下方法：

- 提高PDF渲染分辨率（修改 `scale`参数）
- 确保PDF文件清晰可读
- 选择适合的模型（如paddleocr-vl适合复杂文档，pp-chatocrv4适合信息抽取）
- 使用最新版本的PaddleOCR模型

### 4. 模型下载失败

如果遇到模型下载失败的问题，可以尝试以下方法：

- 检查网络连接是否正常
- 确保有足够的磁盘空间
- 手动下载模型文件并放置到 `.paddlex`目录下

### 5. 日志文件未生成

如果Markdown日志文件未生成，可能的原因：

- 程序没有成功完成任何PDF文件的识别
- 检查当前用户是否有写入文件的权限
- 确保项目目录下有足够的磁盘空间

### 6. 超大图像导致程序无响应

程序已经实现了自动图像缩放功能，会将超过8000像素的图像自动缩放到合适大小。如果仍然遇到问题，可以尝试：

- 手动降低 `max_size`参数值
- 检查PDF文件是否包含异常大的图像
- 使用更高配置的计算机

## 贡献指南

欢迎对项目进行贡献！如果您有任何建议或发现了bug，请通过以下方式参与：

1. Fork项目
2. 创建特性分支：`git checkout -b feature/your-feature`
3. 提交更改：`git commit -am 'Add some feature'`
4. 推送到分支：`git push origin feature/your-feature`
5. 提交Pull Request

## 许可证

本项目采用[Apache License 2.0](LICENSE)许可证。

## 联系方式

如果您有任何问题或建议，可以通过以下方式联系我们：

- 项目地址：https://github.com/prog-le/ppocr_pdf
- 电子邮件：prog.le@outlook.com

## 更新日志

### v1.0.2 (2026-01-22)

- 添加PDF优化功能，支持不同优化级别（low/medium/high）
- 新增灰度渲染选项，减少内存占用，提升处理速度
- 优化了渲染过程，添加动态scale调整
- 实现了更高效的内存管理和资源释放机制
- 为API接口添加了PDF优化参数支持
- 新增Docker部署支持，包括：
  - 多平台镜像构建
  - 非root用户运行
  - 持久化卷挂载
  - 环境变量配置
- 更新了文档，添加了Docker部署指南和API参数说明

### v1.0.1 (2026-01-16)

- 初始版本发布
- 支持PDF文件文字识别
- 实现手动模式和守护模式
- 支持中英文识别
- 自动监控目录变化

---

**感谢使用PaddleOCR PDF文字识别项目！** 🚀
