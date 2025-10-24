# AI6130-NLP

### 环境准备

1. **安装依赖包**
首先需要安装所需的 Python 包：
````bash
pip install fastapi uvicorn chromadb sentence-transformers openai httpx streamlit
````

2. **创建数据目录**
确保项目目录结构如下：
```
RAG/
├── data/
│   ├── normalized/
│   └── chunks.jsonl
├── chroma/
└── scripts/
    ├── app_api.py
    ├── app_streamlit.py
    ├── answer_module.py
    ├── retriever.py
    └── ...
```

### 运行步骤

1. **数据准备和索引构建**
首先需要运行数据处理脚本：
````bash
# 在项目根目录下运行
python scripts/ingest_openfda.py
python scripts/ingest_dailymed.py
python scripts/ingest_pubmed.py
python scripts/preprocess.py
python scripts/build_index.py
````

2. **启动后端服务**
打开一个终端，运行 FastAPI 服务：
````bash
uvicorn scripts.app_api:app --reload --port 8000
````

3. **启动前端界面**
打开另一个终端，运行 Streamlit 应用：
````bash
streamlit run scripts/app_streamlit.py
````

### 使用说明

1. 打开浏览器访问 `http://localhost:8501` 即可看到 Streamlit 界面
2. 在输入框中输入医疗相关的问题
3. 点击提交按钮获取答案和参考文献

### 注意事项

1. 确保已经配置了必要的环境变量：
   - `OPENAI_API_KEY`（如果使用 OpenAI API）
   - `NCBI_API_KEY`（如果使用 PubMed API）
   - `DATA_DIR`（可选，默认为 data）

2. 数据处理和索引构建可能需要一些时间，请耐心等待

3. 首次运行时，模型文件会自动下载，需要稳定的网络连接

4. 如果遇到端口占用，可以修改端口号：
   - FastAPI: `--port 8000`
   - Streamlit: 在命令中添加 `--server.port 8501`
