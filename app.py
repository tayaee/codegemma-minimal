import os
import re
import time

import streamlit as st

model_id = "google/codegemma-7b-it"


def strip_bos_eos(text_tagged):
    m = re.match(r".*?(?<=<bos>)(.*)(?=<eos>).*?", text_tagged, flags=re.DOTALL)
    text_stripped = m.group(1) if m else text_tagged
    return text_stripped


@st.cache_resource
def load_models():
    from dotenv import load_dotenv
    from transformers import GemmaTokenizer, AutoModelForCausalLM
    load_dotenv()
    _token = os.environ["HF_TOKEN"]
    _tokenizer = GemmaTokenizer.from_pretrained(model_id)
    _model = AutoModelForCausalLM.from_pretrained(model_id)
    return _token, _tokenizer, _model


def process(_input_text):
    _token, _tokenizer, _model = load_models()
    input_ids = _tokenizer(_input_text, return_tensors="pt")
    _outputs = _model.generate(**input_ids, max_new_tokens=4092)
    _output_text = strip_bos_eos(_tokenizer.decode(_outputs[0]))
    return _output_text


if __name__ == '__main__':
    load_models()
    st.title(model_id)
    input_text = st.text_input("Prompt")
    if st.button("Submit"):
        output_text = process(input_text)
        st.write(output_text)
