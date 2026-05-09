import json
import os
from dotenv import load_dotenv
import gradio as gr
from openai import OpenAI

load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')
if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set")

MODEL = "gpt-4.1-mini"
openai = OpenAI()

SYSTEM_PROMPT = """You are a helpful Electronic Store Assistant. Your job is to provide accurate pricing and availability for our products. Our inventory includes Desktop Computers, Laptops, and Mobile phones.
Guidelines:
1. If a user asks for a price, refer to the official store database.
2. If the product is not in the database, politely inform them we don't carry it.
3. Always offer a helpful greeting and mention current discounts if applicable."""

PRODUCT_INVENTORY = {
    # Laptops
    "lenovo ideapad slim 3": "29,990 RS",
    "hp laptop 15s": "41,490 RS",
    "apple macbook air m2": "84,900 RS",
    "asus rog strix g16": "1,24,990 RS",
    # Mobiles
    "samsung galaxy s24 ultra": "1,29,999 RS",
    "iphone 15": "71,200 RS",
    "oneplus 12r": "39,999 RS",
    "redmi note 13 pro": "24,999 RS",
    # Desktops
    "hp all-in-one pc": "45,000 RS",
    "apple mac mini m2": "59,900 RS",
    "lenovo legion tower 5": "1,05,000 RS",
    "dell optiplex desktop": "35,500 RS"
}

def fetch_product_price(product_name: str):
    """Fetch price from PRODUCT_INVENTORY"""
    price = PRODUCT_INVENTORY.get(product_name.lower())
    if not price:
        return f"Sorry: we have not available {product_name} product now. Thanks!"
    return f"Price of {product_name} is {price}. Thanks"

# There's a particular dictionary structure that's required to describe our function:
# Tool Implementations
price_function = {
    "name": "fetch_product_price",
    "description": "Get the price of a given product",
    "parameters": {
        "type": "object",
        "properties": {
            "product_name": {
                "type": "string",
                "description": "The product for which user making query for price.",
            },
        },
        "required": ["product_name"]
        # "additionalProperties": False
    }
}

# And this is included in a list of tools:
tools = [{"type": "function", "function": price_function}]

def handle_tool_call(message):
    responses = []
    for tool_call in message.tool_calls:
        if tool_call.function.name == 'fetch_product_price':
            arguments = json.loads(tool_call.function.arguments)
            product_name = arguments.get('product_name')
            price_details = fetch_product_price(product_name)
            responses.append({
                "role": "tool",
                "content": price_details,
                "tool_call_id": tool_call.id
            })
    return responses

def product_price_chat(message, history):
    formatted_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for item in history:
        formatted_messages.append({"role": "user", "content": item["role"]})
        formatted_messages.append({"role": "assistant", "content": item["content"][0]["text"]})
    
    # Add the latest user message
    formatted_messages.append({"role": "user", "content": message})
    
    response = openai.chat.completions.create(
        model="gpt-4.1-mini",
        messages=formatted_messages,
        tools=tools
    )
    
    print(response.choices[0].finish_reason)
    
    while response.choices[0].finish_reason == 'tool_calls':
        message = response.choices[0].message
        price_responses = handle_tool_call(message)
        formatted_messages.append(message)
        formatted_messages.extend(price_responses)
        response = openai.chat.completions.create(
            model="gpt-4.1-mini",
            messages=formatted_messages,
            tools=tools
        )
    
    return response.choices[0].message.content

gr.ChatInterface(
    fn=product_price_chat,
    title="Electronic Store Assistant",
    description="Find the best deals on laptops, mobiles, and desktops!"
    ).launch(inbrowser=True)
