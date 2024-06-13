#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
import sophon.sail as sail
import argparse
import time
# from tokenization_util import make_context
from qwen_generation_utils import make_context
from transformers import AutoTokenizer
import numpy as np
from PIL import Image
import requests
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torch


#convert sail_dtype to numpy dtype
def type_convert(sail_dtype):
    if sail_dtype == sail.Dtype.BM_FLOAT32:
        return np.float32
    if sail_dtype == sail.Dtype.BM_FLOAT16:
        return np.float16
    if sail_dtype == sail.Dtype.BM_INT32:
        return np.int32
    if sail_dtype == sail.Dtype.BM_BFLOAT16: # 后续需要修改bf16的接口,现在先用fp16的代替
        return np.float16
    
    raise TypeError("only support float32 and int32 right now")

def fp16_cast(arr:np.ndarray): #这个接口的作用在于把np.float16假冒成np.uint16传进Tensor，sail update_data如果能接收传输二进制，那就不需要这个了。(后续需要改成bf16的接口)
    """
    reinterpret an array with int16 instead of float16, because pybind11 do not support float16.
    """
    if arr.dtype == np.float16:
        return arr.view(np.uint16)
    else:
        return arr

class Qwen:
    def __init__(self, handles, engine_llm, engine_vit, tokenizer):
        self.handles = handles
        self.sp = tokenizer
        
        # warm up
        self.sp.decode([0]) 
        self.EOS = self.sp.im_end_id

        # load bmodel
        # 这里devio，后面都没有创建系统内存的tensor
        self.net = engine_llm
        self.vit = engine_vit
        self.graph_names = self.net.get_graph_names()

        # initialize qwen parameters
        self.NUM_LAYERS = (len(self.graph_names) - 5) // 2
        self.first_hidden_input_shape = self.net.get_input_shape("block_0", self.net.get_input_names("block_0")[0])
        self.vit_input_shape = self.vit.get_input_shape("qwen_vit", self.vit.get_input_names("qwen_vit")[0])
        _, self.SEQLEN, self.HIDDEN_SIZE = self.first_hidden_input_shape

        # initialize net name
        self.is_greedy_sample = True
        self.name_embed = "embedding"
        self.name_embed_cache = "embedding_cache"
        self.name_lm = "lm_head"
        self.name_blocks = ["block_"+str(i) for i in range(self.NUM_LAYERS)]
        self.name_blocks_cache = ["block_cache_"+str(i) for i in range(self.NUM_LAYERS)]
        self.name_sample = "greedy_head" if self.is_greedy_sample else "penalty_sample_head"
        self.name_vit = "qwen_vit"

        # initialize tensors (inputs & outputs)
        # forward_first: embedding_tensor
        self.first_embed_input = self.init_sail_tensor(self.name_embed, 0, [1, self.SEQLEN])
        self.first_embed_output = self.init_sail_tensor(self.name_embed, 0, [1, self.SEQLEN, self.HIDDEN_SIZE], False)

        # forward_next: embedding_tensor
        self.next_embed_input = self.init_sail_tensor(self.name_embed_cache, 0, [1, 1])
        self.next_embed_output = self.init_sail_tensor(self.name_embed_cache, 0, [1,  self.HIDDEN_SIZE], False)

        # forward_first: hidden_state
        self.first_hidden_input = self.init_sail_tensor(self.name_blocks[0], 0)
        self.first_hidden_output = self.init_sail_tensor(self.name_blocks[0], 0, None, False)

        # forward_next: hidden_state
        self.next_hidden_input = self.init_sail_tensor(self.name_blocks_cache[0], 0)
        self.next_hidden_output = self.init_sail_tensor(self.name_blocks_cache[0], 0, None, False)

        # forward_first: position_id_tensor and attention_mask_tensor
        self.first_pid = self.init_sail_tensor(self.name_blocks[0], 1)
        self.first_attention = self.init_sail_tensor(self.name_blocks[0], 2)
       
        # forward_next: position_id_tensor and attention_mask_tensor
        self.next_pid = self.init_sail_tensor(self.name_blocks_cache[0], 1)
        self.next_attention = self.init_sail_tensor(self.name_blocks_cache[0], 2)

        # forward_next: present_key / present_value (for update kv_cache)
        self.present_key = self.init_sail_tensor(self.name_blocks_cache[0], 1, None, False)
        self.present_value = self.init_sail_tensor(self.name_blocks_cache[0], 2, None, False)

        # forward_first: key_tensor and value_tensor
        self.past_key_output = []
        self.past_value_output = []

        # forward_next: kv cache block 
        self.cache_key_input = []
        self.cache_key_output = []
        self.cache_value_input = []
        self.cache_value_output = []

        for _ in range(self.NUM_LAYERS): 
            self.past_key_output.append(self.init_sail_tensor(self.name_blocks[0], 1, None, False))
            self.past_value_output.append(self.init_sail_tensor(self.name_blocks[0], 2, None, False))

            self.cache_key_input.append(self.init_sail_tensor(self.name_blocks_cache[0], 3))
            self.cache_key_output.append(self.init_sail_tensor(self.name_blocks_cache[0], 1, None, False))

            self.cache_value_input.append(self.init_sail_tensor(self.name_blocks_cache[0], 4))
            self.cache_value_output.append(self.init_sail_tensor(self.name_blocks_cache[0], 2, None, False))

        # lm_head tensor
        self.lm_input = self.init_sail_tensor(self.name_lm, 0)
        self.lm_output = self.init_sail_tensor(self.name_lm, 0, None, False)

        # sample tensor
        self.sample_input = self.init_sail_tensor(self.name_sample, 0)
        self.sample_output = self.init_sail_tensor(self.name_sample, 0, None, False)
        
        # vit tensor
        self.vit_input = self.init_vit_tensor(self.name_vit, 0)
        self.vit_output = self.init_vit_tensor(self.name_vit, 0, None, False)

        self.token_length = 0
    
    def init_vit_tensor(self, name, tensor_idx, shape=None, is_input=True):
        """
        init a sail tensor of sail.engine.
        parameters:
        input:
            name: str, graph_name/net_name
            tensor_idx: int, input/output tensor id
            shape: list[int], shape of tensor
            is_input: bool, is input tensor or not
        return:
            dict
        """
        tensor = {}
        if is_input:
            tensor["name"] = self.vit.get_input_names(name)[tensor_idx]
            tensor["shape"] = self.vit.get_input_shape(name, tensor["name"]) if shape is None else shape
            tensor["dtype"] = self.vit.get_input_dtype(name, tensor["name"])
            tensor["data"] = sail.Tensor(self.handles[1], tensor["shape"], tensor["dtype"], False, True)
        else:
            tensor["name"] = self.vit.get_output_names(name)[tensor_idx]
            tensor["shape"] = self.vit.get_output_shape(name, tensor["name"]) if shape is None else shape
            tensor["dtype"] = self.vit.get_output_dtype(name, tensor["name"])
            tensor["data"] = sail.Tensor(self.handles[1], tensor["shape"], tensor["dtype"], False, True) 
        return tensor

    def init_sail_tensor(self, name, tensor_idx, shape=None, is_input=True):
        """
        init a sail tensor of sail.engine.
        parameters:
        input:
            name: str, graph_name/net_name
            tensor_idx: int, input/output tensor id
            shape: list[int], shape of tensor
            is_input: bool, is input tensor or not
        return:
            dict
        """
        tensor = {}
        if is_input:
            tensor["name"] = self.net.get_input_names(name)[tensor_idx]
            tensor["shape"] = self.net.get_input_shape(name, tensor["name"]) if shape is None else shape
            tensor["dtype"] = self.net.get_input_dtype(name, tensor["name"])
            tensor["data"] = sail.Tensor(self.handles[0], tensor["shape"], tensor["dtype"], False, True)
        else:
            tensor["name"] = self.net.get_output_names(name)[tensor_idx]
            tensor["shape"] = self.net.get_output_shape(name, tensor["name"]) if shape is None else shape
            tensor["dtype"] = self.net.get_output_dtype(name, tensor["name"])
            tensor["data"] = sail.Tensor(self.handles[0], tensor["shape"], tensor["dtype"], False, True) 
        return tensor

    # inference for the first token
    def forward_first(self, token, images):
        input_ids = np.zeros(self.SEQLEN, type_convert(self.first_embed_input["dtype"]))
        input_ids[:min(self.SEQLEN, len(token))] = token
        input_ids = input_ids.reshape(1, -1)
        self.token_length = len(token)
        position_id = np.zeros(self.SEQLEN, type_convert(self.first_pid["dtype"])) 
        for i in range(self.token_length):
            position_id[i] = i

        attention_mask = np.ones(self.SEQLEN*self.SEQLEN, type_convert(self.first_attention["dtype"])) * (-6.109375)
        for i in range(self.token_length):
            for j in range(self.SEQLEN):
                if (j <= i):
                    attention_mask[i*self.SEQLEN + j] = 0
        
        # embedding
        self.first_embed_input["data"].update_data(input_ids)
        input_embed_tensors = {self.first_embed_input["name"]: self.first_embed_input["data"]}
        output_embed_tensors = {self.first_embed_output["name"]: self.first_embed_output["data"]}

        # Embedding Layer Inference
        self.net.process(self.name_embed, input_embed_tensors, output_embed_tensors)

        # ViT Inference
        self.vit_input["data"].update_data(images)
        input_vit_tensors = {self.vit_input["name"]: self.vit_input["data"]}
        output_vit_tensors = {self.vit_output["name"]: self.vit_output["data"]}
        self.vit.process(self.name_vit, input_vit_tensors, output_vit_tensors)

        # blocks
        self.first_hidden_tensor = self.first_embed_output["data"]
        self.first_hidden_tensor.reshape(self.first_hidden_input["shape"])
        self.first_pid["data"].update_data(position_id.reshape(self.first_pid["shape"]))
        # self.first_attention["data"].update_data(fp16_cast(attention_mask.reshape(self.first_attention["shape"]))) # set bf16 in the future.
        self.first_attention["data"].update_data(attention_mask.reshape(self.first_attention["shape"]).view(np.uint16))

        input_blocks_tensors = {self.first_hidden_input["name"]: self.first_hidden_tensor, 
                                self.first_pid["name"]: self.first_pid["data"], 
                                self.first_attention["name"]: self.first_attention["data"]}

        # Transformer Block Inference
        for i in range(self.NUM_LAYERS):        
            output_blocks_tensors = {self.first_hidden_output["name"]: self.first_hidden_tensor,
                                    self.past_key_output[i]["name"]: self.past_key_output[i]["data"],
                                    self.past_value_output[i]["name"]: self.past_value_output[i]["data"]}
            
            self.net.process(self.name_blocks[i], input_blocks_tensors, output_blocks_tensors)
        
        # get the last token info as Lm head input
        copy_len = self.first_hidden_tensor.shape()[-1]
        self.lm_input["data"].sync_d2d(self.first_hidden_tensor,
                                      (self.token_length-1)* copy_len,  
                                      0, 
                                      copy_len)
        
        input_lm_tensors = {self.lm_input["name"]: self.lm_input["data"]}
        output_lm_tensors = {self.lm_output["name"]: self.lm_output["data"]}
        
        # Lm_head Inference
        self.net.process(self.name_lm, input_lm_tensors, output_lm_tensors)
        
        
        # sample
        input_sample_tensor = {self.sample_input["name"]: self.lm_output["data"]}
        output_sample_tensor = {self.sample_output["name"]: self.sample_output["data"]}
        self.net.process(self.name_sample, input_sample_tensor, output_sample_tensor)
        return int(self.sample_output["data"].asnumpy()[0][0])

    # The following tokens prediction
    def forward_next(self, ):
        attention_mask = np.zeros(self.SEQLEN+1, type_convert(self.next_attention["dtype"]))
        for i in range(self.token_length-1, self.SEQLEN):
            attention_mask[i] = -6.109375
        position_id = np.array(self.token_length - 1, type_convert(self.next_pid["dtype"]))

        # embedding
        self.next_embed_input["data"] = self.sample_output["data"]
        self.next_embed_input["data"].reshape(self.next_embed_input["shape"])

        input_embed_tensors = {self.next_embed_input["name"]: self.next_embed_input["data"]}
        output_embed_tensors = {self.next_embed_output["name"]: self.next_embed_output["data"]}
        # Embedding Layer Inference
        self.net.process(self.name_embed_cache, input_embed_tensors, output_embed_tensors)
        
        # blocks
        self.next_pid["data"].update_data(position_id.reshape(self.next_pid["shape"]))
        # self.next_attention["data"].update_data(fp16_cast(attention_mask.reshape(self.next_attention["shape"])))
        self.next_attention["data"].update_data(attention_mask.reshape(self.next_attention["shape"]).view(np.uint16))

        self.next_hidden_tensor = self.next_embed_output["data"]
        self.next_hidden_tensor.reshape(self.next_hidden_input["shape"])

        # Transformer Block Inference
        for i in range(self.NUM_LAYERS):
            inputs_block_cache_tensors = {self.next_hidden_input["name"]: self.next_hidden_tensor, 
                                        self.next_pid["name"]: self.next_pid["data"], 
                                        self.next_attention["name"]: self.next_attention["data"], 
                                        self.cache_key_input[i]["name"]: self.past_key_output[i]["data"], 
                                        self.cache_value_input[i]["name"]: self.past_value_output[i]["data"]}
            outputs_block_cache_tensors = {self.next_hidden_output["name"]: self.next_hidden_tensor,
                                        self.cache_key_output[i]["name"]: self.present_key["data"],
                                        self.cache_value_output[i]["name"]: self.present_value["data"]}
            self.net.process(self.name_blocks_cache[i], inputs_block_cache_tensors, outputs_block_cache_tensors)

            # update kv_cache()
            unit_size = self.present_key["shape"][-1]*self.present_key["shape"][-2]
            self.past_key_output[i]["data"].sync_d2d(self.present_key["data"], 0, (self.token_length-1)*unit_size, unit_size)
            self.past_value_output[i]["data"].sync_d2d(self.present_value["data"], 0, (self.token_length-1)*unit_size, unit_size)
        
        self.lm_input_tensor = self.next_hidden_tensor
        self.lm_input_tensor.reshape(self.lm_input["shape"])
        
        input_lm_tensors = {self.lm_input["name"]: self.lm_input_tensor}
        output_lm_tensors = {self.lm_output["name"]: self.lm_output["data"]}
        
        # Lm_head Inference
        self.net.process(self.name_lm, input_lm_tensors, output_lm_tensors)

        # sample
        input_sample_tensor = {self.sample_input["name"]: self.lm_output["data"]}
        output_sample_tensor = {self.sample_output["name"]: self.sample_output["data"]}
        self.net.process(self.name_sample, input_sample_tensor, output_sample_tensor)
        
        return int(self.sample_output["data"].asnumpy()[0][0])

    def chat_stream(self, input, history, system=''):
        MAX_WINDOW_SIZE = 6144
        CHAT_FORMAT = 'chatml'
        sys_prompt = 'You are a helpful assistant.'
        query = self.sp.from_list_format([
            {'image': 'https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg'},
            {'text': '这是什么'},
        ])
        _, ids = make_context(
            self.sp,
            query,
            history=[],
            system=sys_prompt,
            max_window_size=MAX_WINDOW_SIZE,
            chat_format=CHAT_FORMAT,
        )
        ids_for_pos = torch.unsqueeze(torch.tensor(ids),0)
        bos_pos = torch.where(ids_for_pos == 151857)
        eos_pos = torch.where(ids_for_pos == 151858)
        assert (bos_pos[0] == eos_pos[0]).all()
        img_pos = torch.stack((bos_pos[0], bos_pos[1], eos_pos[1]), dim=1)

        images = []
        image_paths = ['https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg']
        image_size = 448
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
        image_transform = transforms.Compose([
                transforms.Resize(
                    (image_size, image_size),
                    interpolation=InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
        ])
        for image_path in image_paths:
            if image_path.startswith("http://") or image_path.startswith("https://"):
                image = Image.open(requests.get(image_path, stream=True).raw)
            else:
                image = Image.open(image_path)
            image = image.convert("RGB")
            images.append(image_transform(image))
        images = torch.stack(images, dim=0)

        ###
        # query = self.sp.from_list_format([
        #     {'image': 'https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg'},
        #     {'text': '这是什么'},
        # ])
        # input = query
        # ###
        # _, input_tokens = make_context(self.sp, query=input, max_window_size=self.SEQLEN)
        
        # if (len(input_tokens) > self.SEQLEN / 3):
        #     yield '##INPUT_TOO_LONG'
        #     return

        tok_num = 0
        # _, tokens = make_context(self.sp, query=input, 
        #                 history=history,
        #                 system=system,
        #                 max_window_size=self.SEQLEN,
        #                 chat_format="chatml")
        # while (len(tokens) > self.SEQLEN / 2):
        #     if (len(history) > 0):
        #         history = history[1:]
        #     else:
        #         system = ''
        #     _, tokens = make_context(self.sp, query=input, 
        #                     history=history,
        #                     system=system,
        #                     max_window_size=self.SEQLEN,
        #                     chat_format="chatml")
        
        tokens = ids
        first_start = time.time()
        token = self.forward_first(tokens, images)
        first_end = time.time()
        while token != self.EOS and self.token_length < self.SEQLEN:
            diff = self.sp.decode([token])
            yield diff
            if self.token_length < self.SEQLEN:
                self.token_length += 1
            tok_num += 1
            token = self.forward_next()
        
        if self.token_length >= self.SEQLEN:
            yield '##TOKEN_LENGTH_MAX'
            return
        
        next_end = time.time()
        first_duration = first_end-first_start
        next_duration = next_end-first_end
        tps = tok_num / next_duration
        print('\n\n')
        print(f"FTL: {first_duration:.3f} s")
        print(f"TPS: {tps:.3f} token/s")

def app(client):
    history = []
    while True:
        input_str = input("\nQuestion: ")
        if input_str == "exit":
            break
        print("\nAnswer: ")
        assistant_msg = ''
        for response in client.chat_stream(input_str, history):
            assistant_msg = response
            print(response, flush=True, end='')
        history.append([input_str, assistant_msg])

def main(args):
    handles = [sail.Handle(i) for i in args.dev_id]
    tokenizer = AutoTokenizer.from_pretrained(args.token, trust_remote_code=True)
    engine_llm = sail.Engine(args.bmodel, args.dev_id[0], sail.IOMode.DEVIO)
    engine_vit = sail.Engine(args.vitmodel, args.dev_id[1], sail.IOMode.DEVIO)
    client = Qwen(handles, engine_llm, engine_vit, tokenizer)
    app(client)

def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--bmodel', type=str, default='./models/BM1684X/qwen-7b_int4_1dev.bmodel', help='path of bmodel')
    parser.add_argument('--vitmodel', type=str, default='')
    parser.add_argument('--token', type=str, default='./python/token_config/', help='path of tokenizer')
    parser.add_argument('--dev_id', type=list, default=[0, 1], help='dev id')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = argsparser()
    main(args)
    print('all done')