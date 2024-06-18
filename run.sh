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
    pushd models
    # download the models
    popd
    echo "qwenvl_model download!"
else
    echo "$HOME/qwenvl already exist..."
fi


streamlit run python/web_demo.py
