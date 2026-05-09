import gradio as gr
import base64
from io import BytesIO
from PIL import Image
import openai

openai = openai.OpenAI()

def artist(city):
    """Generate a vacation image for the given city"""
    image_response = openai.images.generate(
        model="dall-e-3",
        prompt=f"An image representing a vacation in {city}, showing tourist spots and everything unique about {city}, in a vibrant pop-art style",
        size="1024x1024",
        n=1,
        response_format="b64_json",
    )
    image_base64 = image_response.data[0].b64_json
    image_data = base64.b64decode(image_base64)
    return Image.open(BytesIO(image_data))

def talker(message):
    """Convert text to speech and return as audio file"""
    response = openai.audio.speech.create(
        model="tts-1",
        voice="onyx",
        input=message
    )
    
    # Save to temp file for Gradio
    audio_path = "./tts_output.mp3"
    with open(audio_path, "wb") as f:
        f.write(response.content)
    
    return audio_path

# Create tabbed interface
with gr.Blocks() as demo:
    gr.Markdown("# AI Travel Content Generator")
    
    with gr.Tab("🖼️ Image Generator"):
        with gr.Row():
            city_input = gr.Textbox(
                label="City",
                placeholder="New York City",
                value="New York City"
            )
        generate_img_btn = gr.Button("Generate Image", variant="primary")
        image_output = gr.Image(label="Generated Image", type="pil")
        
        generate_img_btn.click(artist, inputs=city_input, outputs=image_output)
    
    with gr.Tab("🔊 Audio Generator"):
        with gr.Row():
            text_input = gr.Textbox(
                label="Message",
                placeholder="Enter text to convert to speech",
                lines=5,
                value="Welcome to an amazing vacation destination!"
            )
        voice_select = gr.Dropdown(
            choices=["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
            value="onyx",
            label="Voice"
        )
        generate_audio_btn = gr.Button("Generate Audio", variant="primary")
        audio_output = gr.Audio(label="Generated Speech", type="filepath")
        
        generate_audio_btn.click(talker, inputs=text_input, outputs=audio_output)

demo.launch()