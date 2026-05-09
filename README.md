# LLM Engineering - Web Assistant & Store Chat

A comprehensive project demonstrating LLM integration with web scraping, function calling, and interactive chat interfaces using Gradio and Ollama.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Files](#project-files)
- [Screenshots](#screenshots)
- [Requirements](#requirements)

## Overview

This project showcases advanced LLM engineering techniques including:
- Web page content extraction and analysis
- AI-powered chatbots with context awareness
- Function calling for dynamic tool integration
- Multi-store assistant with product inventory management
- Integration with local Ollama models and OpenAI API

## Features

### Page Buddy - Web Assistant
- **Web Scraping**: Extract comprehensive page information (title, headings, links, content, tables)
- **Context-Aware Chat**: Ask questions about any webpage using AI
- **Markdown Export**: Generate detailed markdown reports of web pages
- **Two-Step Interface**: Confirm URL first, then chat with the assistant

### Store Assistant
- **Product Inventory Management**: Maintain database of laptops, mobiles, and desktops
- **Function Calling**: Dynamic tool integration for price lookups
- **Multi-Store Support**: Support for laptop, mobile, and desktop stores
- **Real-time Pricing**: Fetch accurate product prices from inventory

### Store Chat
- **OpenAI Integration**: Direct integration with OpenAI API
- **Tool Calling**: Advanced function calling for product queries
- **Conversational Interface**: Natural language product inquiries
- **Inventory Database**: Comprehensive product catalog with pricing

## Installation

### Prerequisites
- Python 3.8+
- pip (Python package manager)
- Ollama (for local model inference)
- OpenAI API key (for OpenAI integration)

### Step 1: Clone or Download the Project

```bash
git clone <repository-url>
cd llm_engineering
```

### Step 2: Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Set Up Ollama (Optional)

Download and install Ollama from [ollama.ai](https://ollama.ai)

Pull the required model:
```bash
ollama pull llama3.2:1b
```

Start Ollama server:
```bash
ollama serve
```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
OLLAMA_API_KEY=ollama
OLLAMA_BASE_URL=http://localhost:11434/v1
GOOGLE_API_KEY=your_google_api_key_here
```

## Usage

### 1. Page Buddy - Web Assistant

```bash
python page_buddy.py
```

**Features:**
- Enter a webpage URL
- Click "Confirm Link" to validate
- Ask questions about the page content
- Get AI-powered responses based on page context

**Example Questions:**
- "What is the main topic of this page?"
- "Summarize the key points"
- "What links are available on this page?"

### 2. Store Assistant (Ollama)

```bash
python store_assistant.py
```

**Features:**
- Product price lookups using function calling
- Multi-store support (laptops, mobiles, desktops)
- Uses llama3.2:1b model locally

**Example Queries:**
- "What's the price of iPhone 15?"
- "Show me available laptops"
- "Do you have Samsung Galaxy S24?"

### 3. Store Chat (OpenAI)

```bash
python store_chat.py
```

**Features:**
- Advanced function calling with OpenAI
- Real-time product pricing
- Comprehensive inventory management

**Example Queries:**
- "What laptops do you have?"
- "Price of MacBook Air M2?"
- "Available desktops under 50000 RS"

## Project Files

### page_buddy.py
Web assistant application with functions:
- `make_web_request(link)` - Fetch webpage content
- `extract_page_info(link)` - Extract page information
- `create_system_prompt(page_info)` - Generate AI system prompt
- `chat_with_page(web_link, user_message)` - Chat about webpage
- `create_gradio_interface()` - Build Gradio UI

### store_assistant.py
Store assistant with Ollama:
- `fetch_product_price(product_name)` - Look up product prices
- `handle_tool_call(message)` - Process function calls
- `product_price_chat(message, history)` - Chat with tool calling

### store_chat.py
Store chat with OpenAI:
- Same functions as store_assistant.py
- Uses OpenAI API instead of Ollama

## Screenshots

### Page Buddy Interface
![Page Buddy Screenshot](Page%20Buddy%20Screenshot.png)


### Store Assistant Interface
![Store Assistant Screenshot](/Store%20Assistant%20Screenshot.png)


## Requirements

### Python Packages

```
httpx==0.28.1
beautifulsoup4==4.12.3
python-dotenv==1.0.0
openai==1.3.0
gradio==4.26.0
```

### System Requirements

- **RAM**: Minimum 4GB (8GB recommended for Ollama)
- **Storage**: 5GB+ for Ollama models
- **Internet**: Required for OpenAI API, optional for Ollama
- **OS**: Windows, macOS, or Linux

## Troubleshooting

### Ollama Connection Error

**Error**: "Cannot connect to Ollama on localhost:11434"

**Solution**: 
1. Make sure Ollama is installed and running
2. Run `ollama serve` in a terminal
3. Verify the OLLAMA_BASE_URL in .env is correct

### Model Not Found Error

**Error**: "invalid model name"

**Solution**:
1. Pull the model: `ollama pull llama3.2:1b`
2. Verify model is installed: `ollama list`
3. Check model name in code matches exactly

### OpenAI API Key Error

**Error**: "Invalid API key"

**Solution**:
1. Verify your OpenAI API key in .env
2. Check key has not expired
3. Ensure key has proper permissions

## API Integration

### Ollama (OpenAI-Compatible)

```python
from openai import OpenAI

client = OpenAI(
    api_key="ollama",
    base_url="http://localhost:11434/v1"
)

response = client.chat.completions.create(
    model="llama3.2:1b",
    messages=[...]
)
```

### OpenAI Direct

```python
from openai import OpenAI

client = OpenAI(api_key="your_api_key")

response = client.chat.completions.create(
    model="gpt-4-mini",
    messages=[...],
    tools=[...]
)
```

## Key Features Explained

### Web Scraping with BeautifulSoup
- Extracts HTML structure and content
- Parses headings, links, paragraphs, tables
- Generates comprehensive page summaries

### Function Calling
- Dynamic tool integration
- Product price lookups
- Real-time inventory queries
- Extensible tool framework

### Gradio Interface
- User-friendly web UI
- Real-time chat interaction
- Multi-step workflows
- Responsive design
