# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import streamlit as st
from streamlit_server_state import server_state, server_state_lock
from ctransformers import AutoModelForCausalLM as ctAMCL
from PIL import Image 
from transformers import AutoProcessor
from transformers import AutoModelForCausalLM as tAMCL
import re
from os.path import dirname

checkpoint = f'{dirname(__file__)}/mic-git-base/git-base'
llama_checkpoint = f'{dirname(__file__)}/llama-2-ggml/Llama-2-7B-Chat-GGML'

with server_state_lock["processor"]:  # Lock the "count" state for thread-safety
    if "processor" not in server_state:
        server_state.processor = AutoProcessor.from_pretrained(checkpoint)

with server_state_lock["model"]:  # Lock the "count" state for thread-safety
    if "model" not in server_state:
        server_state.model = tAMCL.from_pretrained(checkpoint)
        
def ChatModel(temperature, top_p):
    return ctAMCL.from_pretrained(
        llama_checkpoint, 
        model_type='llama',
        temperature=temperature, 
        top_p = top_p)

with server_state_lock["chat_model"]:  # Lock the "count" state for thread-safety
    if "chat_model" not in server_state:
        server_state.chat_model = ChatModel(0.5, 0.5)
        

st.title(":grey[_AI_]:green[Poet]")

image_file = st.camera_input("Show me a good view that makes me lost") or st.file_uploader("Show me a good picture that ignites my thoughts",type=['png','jpeg','jpg']) 

def load_image(image_file):
    img = Image.open(image_file)
    return img

if image_file is not None:
    file_details = {"FileName":image_file.name,"FileType":image_file.type}
    img = load_image(image_file)
    device = "cpu"
    inputs = server_state.processor(images=img, return_tensors="pt").to(device)
    pixel_values = inputs.pixel_values
    generated_ids = server_state.model.generate(pixel_values=pixel_values, max_length=50)
    generated_caption = server_state.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    Type1 = ''

    type1 = st.radio(
        "You want me to write a long poem?",
        ["Short", "Long"])
    
    Type2 = ''
    
    type2 = st.radio(
        "You want me to write a rhyming poem?",
        ["Rhyming", "Free Verse"])


    if type1 is not None:
        Type1 = type1
        
    if type2 is not None:
        if type2 == "Free Verse":
            type2 = "unrhymed"
        Type2 = type2            




    
    string_dialogue = "You are a creative renowned poet."
    
    if st.button('Leave me to write'):
        with st.spinner("Imagining. I'll take all the time in the world..."):
            output = server_state.chat_model(f"{string_dialogue} Write a {Type1} {Type2} poem on {generated_caption}. Provide the title of the poem at start.")
            output = re.sub(r'([A-Z])', r'\n\1', output)
        with st.container():
            st.image(img, width=400)
            st.write(output)
