import json
import random
import requests
from mtranslate import translate
import streamlit as st

MODEL_URL = "https://api-inference.huggingface.co/models/flax-community/papuGaPT2"
PROMPT_LIST = {
    "Najsmaczniejszy owoc to...": ["Najsmaczniejszy owoc to "],
    "Cześć, mam na imię...": ["Cześć, mam na imię "],
    "Największym polskim poetą był...": ["Największym polskim poetą był "],
}
API_TOKEN = st.secrets["api_token"]
headers = {"Authorization": f"Bearer {API_TOKEN}"}


def query(payload, model_url):
    data = json.dumps(payload)
    print("model url:", model_url)
    response = requests.request("POST", model_url, headers=headers, data=data)
    return json.loads(response.content.decode("utf-8"))


def process(
    text: str, model_name: str, max_len: int, temp: float, top_k: int, top_p: float
):
    payload = {
        "inputs": text,
        "parameters": {
            "max_new_tokens": max_len,
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temp,
            "repetition_penalty": 2.0,
        },
        "options": {
            "use_cache": False,
        },
    }
    return query(payload, model_name)


# Page
st.set_page_config(page_title="papuGaPT2 aka Polish GPT-2 model demo")
st.title("papuGaPT2 aka Polish GPT-2")
# Sidebar
st.sidebar.subheader("Configurable parameters")
max_len = st.sidebar.number_input(
    "Maximum length",
    value=100,
    max_value=250,
    help="The maximum length of the sequence to be generated.",
)
temp = st.sidebar.slider(
    "Temperature",
    value=1.0,
    min_value=0.1,
    max_value=2.0,
    help="The value used to module the next token probabilities.",
)
top_k = st.sidebar.number_input(
    "Top k",
    value=10,
    help="The number of highest probability vocabulary tokens to keep for top-k-filtering.",
)
top_p = st.sidebar.number_input(
    "Top p",
    value=0.95,
    help=" If set to float < 1, only the most probable tokens with probabilities that add up to top_p or higher are kept for generation.",
)
do_sample = st.sidebar.selectbox(
    "Sampling?",
    (True, False),
    help="Whether or not to use sampling; use greedy decoding otherwise.",
)
# Body
st.markdown(
    """
    This demo showcases the text generation capabilities of the papuGaPT2, a GPT2 model pre-trained from scratch using Polish subset of the multilingual Oscar corpus.
    
    To use it, add your text to 'Enter Text' box, or click one of the examples in 'Prompt' drop-down list to load them and click 'Run' button. The model will generate text based on the entered text (prompt).
    
    For more information including dataset, training and evaluation procedure, intended use, limitations and bias analysis see the [model card](https://huggingface.co/flax-community/papuGaPT2).  
    
    In this demo, we used the template from the wonderful demo of [Spanish GPT2 team](https://huggingface.co/spaces/flax-community/spanish-gpt2) especially María Grandury and Manuel Romero.
    """
)
model_name = MODEL_URL
ALL_PROMPTS = list(PROMPT_LIST.keys()) + ["Custom"]
prompt = st.selectbox("Prompt", ALL_PROMPTS, index=len(ALL_PROMPTS) - 1)
if prompt == "Custom":
    prompt_box = "Enter your text here"
else:
    prompt_box = random.choice(PROMPT_LIST[prompt])
text = st.text_area("Enter text", prompt_box)
if st.button("Run"):
    with st.spinner(text="Getting results..."):
        st.subheader("Result")
        print(f"maxlen:{max_len}, temp:{temp}, top_k:{top_k}, top_p:{top_p}")
        result = process(
            text=text,
            model_name=model_name,
            max_len=int(max_len),
            temp=temp,
            top_k=int(top_k),
            top_p=float(top_p),
        )
        print("result:", result)
        if "error" in result:
            if type(result["error"]) is str:
                st.write(f'{result["error"]}.', end=" ")
                if "estimated_time" in result:
                    st.write(
                        f'Please try again in about {result["estimated_time"]:.0f} seconds.'
                    )
            elif type(result["error"]) is list:
                for error in result["error"]:
                    st.write(f"{error}")
        else:
            result = result[0]["generated_text"]
            st.write(result.replace("\n", "  \n"))
            st.text("English translation")
            st.write(translate(result, "en", "pl").replace("\n", "  \n"))
