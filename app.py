from gpt_index import SimpleDirectoryReader, GPTListIndex, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain.chat_models import ChatOpenAI
import gradio as gr
import config
import sys
import os

os.environ["OPENAI_API_KEY"]=config.key

def construct_index(directory_path):
    max_input_size = 4096
    num_outputs = 512
    max_chunk_overlap = 20
    chunk_size_limit = 600

    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo", max_tokens=num_outputs))

    documents = SimpleDirectoryReader(directory_path).load_data()

    index = GPTSimpleVectorIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    index.save_to_disk('index.json')

    return index

def chatbot(input_text):
    index = GPTSimpleVectorIndex.load_from_disk('index.json')
    response = index.query(input_text, response_mode="default")
    return response.response

iface = gr.Interface(fn=chatbot,
                     inputs=gr.components.Textbox(lines=7, label="Enter your query here"),
                     outputs="text",
                     title="FamCart AI Chatbot",
                     live=True,
                     theme=gr.themes.Soft(),
                     description="Your personalized AI chatbot for Financial Literacy , Management and FamCart!",
                     examples=[["Explain me about Financial Literacy."], ["How can I save money?"],
                               ["How can I use FamCart?"], ["What is FamCart?"]],
                     allow_flagging = "never",
                
                     )

index = construct_index("docs")
iface.launch(share=True)