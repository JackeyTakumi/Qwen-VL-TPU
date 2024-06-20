#!/bin/bash
set -ex

res=$(which unzip)

if [ $? != 0 ];
then
    echo "Please install unzip on your system!"
    exit
fi

pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade

# download bert_model
if [ ! -d "./models" ]; then
    echo "./models does not exist, download..."
    mkdir -p ./models
    pushd models
    # download the models
    python3 -m dfss --url=open@sophgo.com:ext_model_information/LLM/LLM-TPU/qwen-vl-chat_int8_seq1024_1dev_f16.bmodel
    python3 -m dfss --url=open@sophgo.com:ext_model_information/LLM/LLM-TPU/qwen_vit_1684x_f16.bmodel
    popd
    echo "qwenvl_model download!"
else
    echo "$HOME/qwenvl already exist..."
fi


streamlit run python/web_demo.py
