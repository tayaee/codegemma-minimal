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
    print(f'Using HF_TOKEN={_token[:8]}{"*" * (len(_token) - 8)}')
    t1 = round(time.time(), 1)
    print(f'Loading tokenizer...')
    _tokenizer = GemmaTokenizer.from_pretrained(model_id)
    t2 = round(time.time(), 1)
    print(f'Loaded tokenizer {model_id}, {round(t2 - t1, 1)} secs. Loading model...')
    _model = AutoModelForCausalLM.from_pretrained(model_id)
    t3 = round(time.time(), 1)
    print(f'Loaded model {model_id}, {round(t3 - t2, 1)} secs.')
    return _token, _tokenizer, _model


def process(_input_text):
    _token, _tokenizer, _model = load_models()
    input_ids = _tokenizer(_input_text, return_tensors="pt")
    t1 = round(time.time(), 1)
    print(f'Generating output...')
    _outputs = _model.generate(**input_ids, max_new_tokens=4092)
    _output_text = strip_bos_eos(_tokenizer.decode(_outputs[0]))
    t2 = round(time.time(), 1)
    print(f'Generated output: {_output_text}, {round(t2 - t1, 1)} secs')
    return _output_text


if __name__ == '__main__':
    load_models()
    st.title(model_id)
    input_text = st.text_input("Prompt")
    if st.button("Submit"):
        output_text = process(input_text)
        st.write(output_text)
