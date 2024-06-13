import streamlit as st
import sophon.sail as sail
from qwen_vl import Qwen
from transformers import AutoTokenizer
from PIL import Image

token_path = './supports/token_config'
bmodel_path = '../../qwen-vl-chat_int4_seq512_1dev.bmodel'
vit_path = '../../qwen_vit_1684x_bf16.bmodel'
dev_id = [0,1]

st.title("Qwen-VL")

# Function to display uploaded image in the sidebar
def display_uploaded_image(image):
    st.sidebar.image(image, caption='Uploaded Image', use_column_width=True)

uploaded_file = st.file_uploader("上传图片", type=["jpg", "jpeg", "png"])
# print(type(uploaded_file))

# Check if a file was uploaded
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    print(type(image))

    # Display the uploaded image in the sidebar
    display_uploaded_image(uploaded_file)

    @st.cache_resource
    def get_handles():
        return [sail.Handle(i) for i in dev_id]

    @st.cache_resource
    def get_vit():
        return sail.Engine(vit_path, dev_id[1], sail.IOMode.DEVIO)

    @st.cache_resource
    def get_engine():
        return sail.Engine(bmodel_path, dev_id[0], sail.IOMode.DEVIO)

    @st.cache_resource
    def get_tokenizer():
        return AutoTokenizer.from_pretrained(token_path, trust_remote_code=True)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Initialize sail.Handle
    if "handles" not in st.session_state:
        st.session_state.handles = get_handles()
    
    # Initialize sail.Engine
    if "engine" not in st.session_state:
        st.session_state.engine = get_engine()

    if "vit" not in st.session_state:
        st.session_state.vit = get_vit()

    # Initialize Tokenizer
    if "tokenizer" not in st.session_state:
        st.session_state.tokenizer = get_tokenizer()

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("请输入您的问题 "):
        # Display user message in chat message container
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        client = Qwen(st.session_state.handles, st.session_state.engine, st.session_state.vit, st.session_state.tokenizer)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
        
            stream = client.chat_stream(input=prompt, history=[[m["role"], m["content"]] for m in st.session_state.messages])
            response = st.write_stream(stream)

            # Add assistant message to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
