import httpx
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from datetime import datetime
import os
from dotenv import load_dotenv
import gradio as gr
from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize OpenAI client with Ollama base URL
ollama_base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434/v1')
ollama_api_key = os.getenv('OLLAMA_API_KEY', 'ollama')

client = OpenAI(
    api_key=ollama_api_key,
    base_url=ollama_base_url
)


def make_web_request(link: str) -> dict:
    """
    Make a web request to the given URL and return the content and status code.
    
    Args:
        link: The URL to request
        
    Returns:
        A dictionary containing:
            - 'status_code': HTTP status code (int)
            - 'content': Response body as text (str)
    """
    try:
        response = httpx.get(link, timeout=10.0)
        return {
            'status_code': response.status_code,
            'content': response.text
        }
    except httpx.RequestError as e:
        return {
            'status_code': None,
            'content': f"Error: {str(e)}"
        }


def extract_page_info(link: str) -> dict:
    """
    Extract detailed information from a webpage including title, body, links, headings, etc.
    
    Args:
        link: The URL to extract information from
        
    Returns:
        A dictionary containing:
            - 'url': The original URL
            - 'title': Page title
            - 'description': Meta description
            - 'headings': All headings (h1-h6)
            - 'body': Main body text
            - 'links': All links with text and href
            - 'paragraphs': All paragraphs
            - 'lists': All list items
            - 'tables': Table data if present
            - 'status_code': HTTP status code
    """
    try:
        # Use make_web_request to fetch the page
        response_data = make_web_request(link)
        
        if response_data['status_code'] is None:
            return {
                'url': link,
                'error': response_data['content'],
                'status_code': None
            }
        
        soup = BeautifulSoup(response_data['content'], 'html.parser')
        
        # Extract title
        title = soup.title.string if soup.title else "No title found"
        
        # Extract meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        description = meta_desc['content'] if meta_desc else "No description found"
        
        # Extract all headings
        headings = {}
        for i in range(1, 7):
            tag = f'h{i}'
            headings[tag] = [h.get_text(strip=True) for h in soup.find_all(tag)]
        
        # Extract body text
        body_text = soup.get_text(separator='\n', strip=True)
        
        # Extract all links
        links = []
        for a in soup.find_all('a', href=True):
            links.append({
                'text': a.get_text(strip=True),
                'href': urljoin(link, a['href']),
                'title': a.get('title', '')
            })
        
        # Extract all paragraphs
        paragraphs = [p.get_text(strip=True) for p in soup.find_all('p')]
        
        # Extract all list items
        list_items = [li.get_text(strip=True) for li in soup.find_all('li')]
        
        # Extract table data
        tables = []
        for table in soup.find_all('table'):
            table_data = []
            for row in table.find_all('tr'):
                cells = [cell.get_text(strip=True) for cell in row.find_all(['td', 'th'])]
                table_data.append(cells)
            tables.append(table_data)
        
        return {
            'url': link,
            'title': title,
            'description': description,
            'headings': headings,
            'body': body_text,
            'links': links,
            'paragraphs': paragraphs,
            'lists': list_items,
            'tables': tables,
            'status_code': response_data['status_code']
        }
    
    except Exception as e:
        return {
            'url': link,
            'error': str(e),
            'status_code': None
        }


def create_system_prompt(page_info: dict) -> str:
    """
    Create a system prompt for the chat assistant using page information as context.
    
    Args:
        page_info: Dictionary containing extracted page information
        
    Returns:
        A formatted system prompt string
    """
    # Format page content for context
    headings_text = ""
    for level in range(1, 7):
        tag = f'h{level}'
        if page_info['headings'][tag]:
            headings_text += f"\n{tag.upper()} Headings:\n"
            for heading in page_info['headings'][tag]:
                headings_text += f"- {heading}\n"
    
    links_text = ""
    if page_info['links']:
        links_text = "\nAvailable Links:\n"
        for link_item in page_info['links'][:10]:  # Limit to first 10 links
            links_text += f"- {link_item['text']}: {link_item['href']}\n"
    
    system_prompt = f"""You are a helpful web assistant responsible for answering user questions about the following webpage.

**Page Information:**
- URL: {page_info['url']}
- Title: {page_info['title']}
- Description: {page_info['description']}

**Page Structure:**
{headings_text}

**Page Content:**
{page_info['body'][:2000]}...

**Available Links:**
{links_text}

**Instructions:**
1. Answer user questions based on the page content provided above
2. Be helpful, accurate, and concise
3. If the user asks about something not on the page, politely let them know
4. Provide relevant links when appropriate
5. Maintain a friendly and professional tone
6. If you don't have enough information to answer, suggest what additional information would help"""
    
    return system_prompt


def chat_with_page(web_link: str, user_message: str) -> str:
    """
    Chat with Ollama model about a webpage using OpenAI client.
    
    Args:
        web_link: The URL of the webpage to discuss
        user_message: The user's question or message
        
    Returns:
        The assistant's response
    """
    try:
        # Extract page information
        page_info = extract_page_info(web_link)
        
        if 'error' in page_info:
            return f"Error extracting page: {page_info['error']}"
        
        # Create system prompt with page context
        system_prompt = create_system_prompt(page_info)
        
        # Call Ollama API using OpenAI client
        response = client.chat.completions.create(
            model="llama3.2:1b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        return f"Error: {str(e)}"


def create_gradio_interface():
    """
    Create a Gradio interface for the web assistant chatbot.
    
    Returns:
        Gradio Interface object
    """
    
    # Store the current web link
    current_link = {"value": None}
    
    def confirm_link(web_link: str) -> str:
        """
        Confirm and store the web link.
        
        Args:
            web_link: The webpage URL
            
        Returns:
            Confirmation message
        """
        if not web_link:
            return "Please enter a valid web link"
        
        current_link["value"] = web_link
        return f"Link confirmed: {web_link}\n\nYou can now ask questions about this page in the chat below."
    
    def chat_function(user_message: str, chat_history: list = None) -> str:
        """
        Chat function for gr.ChatInterface.
        
        Args:
            user_message: User's message
            chat_history: Previous chat messages
            
        Returns:
            Assistant's response
        """
        if current_link["value"] is None:
            return "Please confirm a web link first before chatting."
        
        if not user_message:
            return "Please enter a message"
        
        # Get response from chat
        response = chat_with_page(current_link["value"], user_message)
        return response
    
    # Create the interface
    with gr.Blocks(title="Page Buddy - Web Assistant", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# Page Buddy - Web Assistant")
        gr.Markdown("Enter a webpage URL, confirm it, and then ask questions about the page content.")
        
        # Link input section
        with gr.Group():
            gr.Markdown("## Step 1: Enter Webpage URL")
            with gr.Row():
                web_link_input = gr.Textbox(
                    label="Webpage URL",
                    placeholder="https://example.com",
                    lines=1
                )
            
            with gr.Row():
                confirm_btn = gr.Button("Confirm Link", variant="primary", size="lg")
            
            confirmation_output = gr.Textbox(
                label="Status",
                interactive=False,
                lines=2
            )
        
        gr.Markdown("---")
        
        # Chat section
        with gr.Group():
            gr.Markdown("## Step 2: Chat with Assistant")
            chatbot = gr.ChatInterface(
                fn=chat_function,
                examples=[
                    "What is the main topic of this page?",
                    "Summarize the key points",
                    "What links are available on this page?",
                    "Tell me more about the content"
                ],
                title="Chat with Page Assistant",
                description="Ask questions about the webpage content"
            )
        
        # Handle confirm button
        confirm_btn.click(
            fn=confirm_link,
            inputs=[web_link_input],
            outputs=[confirmation_output]
        )
    
    return interface


# Run the Gradio interface
if __name__ == "__main__":
    interface = create_gradio_interface()
    interface.launch(inbrowser=True)
