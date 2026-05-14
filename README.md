# AI 校规问答系统（混合模式 RAG + 本地 Ollama 模型）

本项目是一个 **混合 RAG + 本地 Ollama 模型** 的学校行为规范问答系统。  
支持单条终端问答，并保留了缓存机制与上下文长度优化。

## 功能特点

- **RAG 检索**：根据输入问题检索最相关的校规条款
- **本地大模型判断**：结合 Ollama 本地模型 `deepseek-r1:8b` 做最终判断
- **缓存机制**：同一个问题重复查询时直接使用缓存，提高效率
- **上下文截断优化**：限制检索上下文总长度，避免超长输入
- **终端问答**：无需网络即可直接在 CMD / VSCode 终端运行
- **简洁输出**：直接输出 `允许/不允许` 和简短理由

## Ollama 与模型准备

- **01.下载 Ollama 官方程序并安装**:
安装完成后记下路径，例如 Windows 默认路径：
C:/Users/你的用户名/AppData/Local/Programs/Ollama/ollama.exe

- **下载本地模型 deepseek-r1:8b**：
在终端中执行 ollama pull deepseek-r1:8b

- **在代码中修改 Ollama 路径和模型名称**：
OLLAMA_PATH = "你的 Ollama 安装路径/ollama.exe"
MODEL_NAME = "deepseek-r1:8b"

## 环境依赖

使用 Python 3.11+，安装依赖：
在终端中执行 pip install -r requirements.txt

