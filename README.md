# 🌊 YAMUNA MAS — Multi-Agent AI System

**YAMUNA MAS (Multi-Agent System)** is an AI-powered multi-agent platform that integrates multiple intelligent agents, AI/LLM capabilities, Model Context Protocol (MCP), Gradio-based user interaction, and server-side communication into a unified system.

The project is designed as an experimental multi-agent AI platform capable of coordinating different agents and services through a centralized interface.

---

## 🚀 Overview

YAMUNA MAS v3.0 is a multi-agent artificial intelligence platform developed using Python and modern AI technologies.

The system combines:

* 🤖 Multi-agent architecture
* 🧠 AI/LLM-powered processing
* 🔌 Model Context Protocol (MCP)
* 🌐 Gradio web interface
* 📡 MCP SSE server
* ⚡ Parallel agent/service execution
* 📊 Performance benchmarking
* 🛠️ Extensible agent and engine architecture

The project is implemented primarily as a Jupyter Notebook and provides an interactive environment for running and testing the complete YAMUNA platform.

---

## ✨ Key Features

### 🤖 Multi-Agent Architecture

YAMUNA MAS is structured around multiple agents and processing engines that can work together as part of a larger AI workflow.

The multi-agent approach allows different components to handle different tasks while contributing to the overall system.

### 🧠 AI & LLM Integration

The platform integrates Large Language Model capabilities for intelligent processing and agent-based operations.

The project uses the **Groq API** for LLM-related functionality.

> **Security:** API credentials must never be committed directly to GitHub.

### 🔌 Model Context Protocol (MCP)

YAMUNA MAS incorporates the **Model Context Protocol (MCP)** to provide a standardized way for AI systems and tools to communicate with external services and capabilities.

The project includes MCP-related server functionality and supports **MCP SSE communication**.

### 🌐 Gradio Interface

A Gradio-based interface provides an interactive way to access and test the platform.

This makes it possible to interact with the system without requiring a separate frontend application.

### 📡 MCP SSE Server

The project includes functionality for launching an MCP server using **Server-Sent Events (SSE)**.

This enables communication between the MCP components and connected clients.

### ⚡ Dual Launch

YAMUNA MAS provides a dual-launch setup for running:

* Gradio interface
* MCP SSE server

This allows the user interface and MCP services to operate together.

### 📊 Performance Benchmarking

The notebook includes performance-testing and benchmarking functionality to evaluate the behavior and execution performance of the platform.

---

## 🏗️ System Architecture

The overall architecture can be represented as:

```text
                    ┌──────────────────────┐
                    │      User / Client   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Gradio Interface   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │    YAMUNA MAS Core   │
                    └──────────┬───────────┘
                               │
                ┌──────────────┼──────────────┐
                ▼              ▼              ▼
          ┌──────────┐   ┌──────────┐   ┌──────────┐
          │  Agent 1 │   │  Agent 2 │   │  Agent N │
          └────┬─────┘   └────┬─────┘   └────┬─────┘
               │              │              │
               └──────────────┼──────────────┘
                              ▼
                    ┌──────────────────────┐
                    │    AI / LLM Layer    │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │      MCP Layer       │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │     MCP SSE Server   │
                    └──────────────────────┘
```

---

## 🛠️ Technologies Used

| Technology           | Purpose                                      |
| -------------------- | -------------------------------------------- |
| **Python**           | Core programming language                    |
| **Jupyter Notebook** | Interactive development and execution        |
| **Gradio**           | Web-based user interface                     |
| **Groq API**         | Large Language Model integration             |
| **MCP**              | Model Context Protocol integration           |
| **MCP SSE**          | Server-Sent Events communication             |
| **nest-asyncio**     | Async execution inside notebook environments |
| **python-docx**      | Document generation/processing               |
| **localtunnel**      | External access/tunneling during development |

---

## 📁 Project Structure

```text
YAMUNA-MAS/
│
├── Yamuna_MAS_MCP.ipynb
│
├── .gitignore
│
└── README.md
```

The main application and platform implementation are contained within the Jupyter Notebook.

---

## 💻 Requirements

Before running the project, make sure the following are installed:

* Python 3.x
* Jupyter Notebook or JupyterLab
* Internet connection
* Required Python packages
* A valid Groq API key

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Aryan091203/YAMUNA-MAS.git
```

### 2. Navigate to the Project

```bash
cd YAMUNA-MAS
```

### 3. Install Required Packages

Run the following in your Python environment:

```bash
pip install gradio groq python-docx mcp nest-asyncio
```

If MCP CLI functionality is required:

```bash
pip install "mcp[cli]"
```

For local tunneling:

```bash
npm install -g localtunnel
```

---

## 🔐 API Key Configuration

The project uses the Groq API for AI/LLM functionality.

**Never place your real API key directly inside the notebook before uploading the project to GitHub.**

Instead, configure it using an environment variable.

### Windows PowerShell

```powershell
$env:GROQ_API_KEY="your_api_key_here"
```

### Python

```python
import os
from groq import Groq

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=GROQ_API_KEY)
```

For permanent local configuration, an environment file can also be used, provided that `.env` is included in `.gitignore`.

---

## ▶️ Running the Project

Open the notebook:

```text
Yamuna_MAS_MCP.ipynb
```

Then execute the cells in their intended order.

The notebook is organized into multiple stages, including:

1. Environment and package setup
2. YAMUNA platform initialization
3. Agent and engine configuration
4. Gradio interface setup
5. MCP server setup
6. Dual launch of services
7. AI/LLM interaction
8. Performance benchmarking

---

## 🌐 Gradio Interface

The Gradio component provides a browser-based interface for interacting with the YAMUNA platform.

After successful execution, the notebook provides the required local interface information.

Open the generated Gradio URL in a web browser to interact with the application.

---

## 🔌 MCP Integration

YAMUNA MAS integrates the **Model Context Protocol (MCP)** to provide structured communication between the AI system and MCP-enabled tools/services.

The platform includes MCP server functionality and supports SSE-based communication.

Conceptually:

```text
AI Agents
    │
    ▼
YAMUNA MAS
    │
    ▼
MCP
    │
    ▼
MCP SSE Server
    │
    ▼
Connected Tools / Services
```

---

## 📡 Dual Service Launch

One of the important components of the platform is its dual-launch capability.

The system can operate with:

```text
┌─────────────────┐
│ Gradio Web UI   │
└────────┬────────┘
         │
         │
┌────────▼────────┐
│  YAMUNA MAS     │
└────────┬────────┘
         │
┌────────▼────────┐
│ MCP SSE Server  │
└─────────────────┘
```

This allows the interactive interface and MCP communication layer to be used together.

---

## 📊 Performance Testing

The project contains benchmarking functionality for evaluating the execution and performance of the platform.

Benchmarking can be used to study:

* Agent execution performance
* Processing time
* Service response behavior
* Overall system performance

The benchmark results are generated during notebook execution.

---

## 🔄 Project Workflow

The general workflow of YAMUNA MAS is:

```text
Start
  │
  ▼
Initialize Environment
  │
  ▼
Load Required Libraries
  │
  ▼
Initialize YAMUNA Platform
  │
  ▼
Initialize Agents & Engines
  │
  ▼
Configure AI / LLM
  │
  ▼
Launch Gradio Interface
  │
  ▼
Launch MCP SSE Server
  │
  ▼
Process Requests
  │
  ▼
Execute Agents / Tools
  │
  ▼
Return Results
  │
  ▼
Performance Evaluation
```

---

## 🔒 Security Considerations

This project uses external AI services and therefore requires API credentials.

For security:

* Never commit API keys to GitHub.
* Never share private API credentials publicly.
* Store secrets using environment variables.
* Keep `.env` files out of version control.
* Rotate/revoke any API key that has accidentally been exposed.
* Do not include sensitive credentials in Jupyter notebook cells.

A `.gitignore` file should include:

```gitignore
.env
__pycache__/
.ipynb_checkpoints/
.venv/
venv/
.vscode/
```

---

## 🧪 Development

YAMUNA MAS is suitable for experimentation and development in:

* Multi-agent AI
* LLM applications
* AI agent orchestration
* MCP-based systems
* AI tool integration
* Interactive AI interfaces
* AI performance evaluation

The notebook-based structure also makes it convenient to experiment with individual components and workflows.

---

## 🔮 Future Improvements

Potential future improvements include:

* Modularizing agents into separate Python modules
* Adding a dedicated frontend
* Adding persistent configuration management
* Improving error handling
* Adding automated tests
* Adding structured logging
* Improving agent-to-agent communication
* Adding more MCP-compatible tools
* Adding authentication and access control
* Containerizing the application with Docker
* Deploying the platform to a cloud environment
* Creating a production-ready API layer
* Adding detailed performance dashboards

---

## 🎯 Project Objectives

The main objectives of YAMUNA MAS are to explore and demonstrate:

1. Multi-agent artificial intelligence
2. LLM-powered agent workflows
3. Model Context Protocol integration
4. AI tool/service communication
5. Interactive AI applications
6. MCP SSE-based communication
7. Performance evaluation of AI systems

---

## 📚 Academic / Research Purpose

YAMUNA MAS can be used as an educational and experimental project for understanding modern AI system architecture.

It demonstrates how multiple AI components, agents, interfaces, and communication protocols can be combined into a single platform.

---

## 👨‍💻 Author

**Aryan**

YAMUNA MAS v3.0

---

## 🌐 Repository

**GitHub:**
https://github.com/Aryan091203/YAMUNA-MAS

---

## 🤝 Contributing

Contributions and improvements are welcome.

To contribute:

1. Fork the repository.
2. Create a new branch.
3. Make your changes.
4. Test the changes.
5. Commit your work.
6. Open a Pull Request.

---

## ⚠️ Disclaimer

This project is intended for educational, experimental, and research purposes.

Users are responsible for configuring their own API credentials securely and for complying with the terms and policies of any external AI services used by the project.

---

## ⭐ Support

If you find this project useful, consider giving the repository a ⭐ on GitHub.

---

## 📄 License

This project does not currently specify a separate open-source license. Unless a license is added to the repository, the project should be treated as **all rights reserved** by the author.
