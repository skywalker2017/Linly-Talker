import gradio as gr
import openai

# Set your OpenAI API key
openai.api_key = "YOUR_API_KEY"

def ask(question):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": question}],
        temperature=0.7
    )
    return response['choices'][0]['message']['content']

with gr.Blocks() as demo:
    gr.Markdown("# LLM Question Answering")
    
    with gr.Row():
        model_input = gr.Textbox(label="Your Question:", placeholder="Type your question here...", interactive=True)
        ask_button = gr.Button("Ask")
        
    model_output = gr.Textbox(label="The Answer:", interactive=False)
    
    ask_button.click(ask, inputs=model_input, outputs=model_output)

demo.launch()