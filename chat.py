# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from llama import Llama

import argparse
from pathlib import Path
import os
import time
import torch
import random
import numpy as np

WORKER_JOB_TYPE = "llama3"
DEFAULT_WORKER_AUTH_KEY = "5b07e305b50505ca2b3284b4ae5f65d1"
VERSION = 0

def main():
    """
    Entry point of the program for generating text using a pretrained model.

    Args:
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 512.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 8.
        max_gen_len (int, optional): The maximum length of generated sequences. If None, it will be
            set to the model's max sequence length. Defaults to None.
    """
    args = load_flags()
    if not args.tokenizer_path:
        args.tokenizer_path = str(Path(args.ckpt_dir) / 'tokenizer.model')
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if args.api_server:

        from aime_api_worker_interface import APIWorkerInterface

        api_worker = APIWorkerInterface(args.api_server, WORKER_JOB_TYPE, args.api_auth_key, args.gpu_id, world_size=world_size, rank=local_rank, gpu_name=torch.cuda.get_device_name(), worker_version=VERSION)
        callback = ProcessOutputCallback(local_rank, api_worker, Path(args.ckpt_dir).name, args.max_seq_len)

    generator = Llama.build(
        ckpt_dir=args.ckpt_dir,
        tokenizer_path=args.tokenizer_path,
        max_seq_len=args.max_seq_len,
        max_batch_size=args.max_batch_size,
    )
    set_seed(args.seed)
    if args.api_server:
        while True:
            prompts = []
            
            job_batch_data = api_worker.job_batch_request(args.max_batch_size)
            batch_size = [len(job_batch_data)]
            torch.distributed.broadcast_object_list(batch_size, 0)
            batch_size = batch_size[0]

            if local_rank == 0:
                print(f'processing job ', end='', flush=True)
                for job_data in job_batch_data:
                    print(f'{job_data.get("job_id")} ... ', end='', flush=True)
                    
                    prompt_input = job_data.get('prompt_input')
                    if prompt_input is None:
                        prompt_input = job_data.get('text')
                    chat_context = job_data.get('chat_context')
                    if chat_context:
                        chat_context.append(
                            {
                                "role": "user", 
                                "content": prompt_input
                            }
                        )
                        prompts.append(chat_context)
                    else:
                        prompts.append(prompt_input)
                top_ps = api_worker.get_job_batch_parameter('top_p')
                top_ks = api_worker.get_job_batch_parameter('top_k')
                temperatures = api_worker.get_job_batch_parameter('temperature')
                max_gen_tokens = api_worker.get_job_batch_parameter('max_gen_tokens')
                if None in max_gen_tokens:
                    max_gen_tokens = [500 for _ in range(batch_size)]
            else:
                prompts = ['' for _ in range(batch_size)] # array has to be same size for multi rank broadcast
                top_ps = [args.top_p for _ in range(batch_size)] # array has to be same size for multi rank broadcast
                top_ks = [args.top_k for _ in range(batch_size)] # array has to be same size for multi rank broadcast
                temperatures = [args.temperature for _ in range(batch_size)] # array has to be same size for multi rank broadcast
                max_gen_tokens = [500 for _ in range(batch_size)]

            # synchronize across ranks
            torch.distributed.broadcast_object_list(prompts, 0)
            torch.distributed.broadcast_object_list(top_ps, 0)
            torch.distributed.broadcast_object_list(top_ks, 0)
            torch.distributed.broadcast_object_list(temperatures, 0)
            torch.distributed.broadcast_object_list(max_gen_tokens, 0)

            generator.generate_realtime(
                callback, prompts, max_gen_len=max_gen_tokens, temperatures=temperatures, top_ps=top_ps, top_ks=top_ks
            )

            print('Done')
    else:
        if not args.temperature:
            args.temperature = 0.8
        ctx = [
            {
                "role": "system",
                "content": 
                    "You are a helpful, respectful and honest assistant named Steve. " +\
                    "Always answer as helpfully as possible, while being safe. " +\
                    "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. " +\
                    "Please ensure that your responses are socially unbiased and positive in nature. " +\
                    "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. " +\
                    "If you don't know the answer to a question, please don't share false information."
            },
            {
                "role": "user", 
                "content": "Hello, Steve."
            },
            {
                "role": "assistant", 
                "content": "How can I assist you today?"
            },
        ]
        print(f'\n{ctx[0]["content"]}')
        print('User: ', f'{ctx[1]["content"]}')
        print('Steve: ',f'{ctx[2]["content"]}')
        callback = ProcessOutputToShellCallback(local_rank, ctx)
        while True:
            if local_rank == 0:
                prompt = input(f'User: ')
              
                print("Steve: ", end='', flush=True)
                callback.ctx.append(
                    {
                        "role": "user", 
                        "content": prompt
                    }
                )
                
                prompts = [callback.ctx]
            else:
                prompts = ['']
            torch.distributed.broadcast_object_list(prompts, src=0)
            if not args.temperature:
                args.temperature = 0.8
            if not args.top_p:
                args.top_p = 0.9
            if not args.top_k:
                args.top_k = 40
            generator.generate_realtime(
                callback, prompts, max_gen_len=[1024], temperatures=[args.temperature], top_ps=[args.top_p], top_ks=[args.top_k]
            )



def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def load_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_dir", type=str, required=False,
        help="Location of LLama weights",
    )
    parser.add_argument(
        "--tokenizer_path", type=str, required=False,
        help="Location of tokenizer"
    )
    parser.add_argument(
        '--temperature', type=float, required=False,
        help='Temperature'
    )
    parser.add_argument(
        "--top_p", type=float, required=False,
        help="Top_p, 0=<top_p<=1"
    )
    parser.add_argument(
        "--top_k", type=int, required=False,
        help="Top_k, 0=<top_k<=1",
    )
    parser.add_argument(
        "--max_seq_len", type=int, default=8192, required=False,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--max_batch_size", type=int, default=1, required=False,
        help="Maximum batch size",
    )    
    parser.add_argument(
        "--seed", type=int, default=1234, required=False,
        help="Initial Seed",
    )    
    parser.add_argument(
        "--repetition_penalty", type=float, default=(1.0/0.85), required=False,
        help="Repetition penalty",
    )
    parser.add_argument(
        "--api_server", type=str, required=False,
        help="Address of the API server"
    )
    parser.add_argument(
        "--gpu_id", type=int, default=0, required=False,
        help="ID of the GPU to be used"
    )
    parser.add_argument(
        "--api_auth_key", type=str , default=DEFAULT_WORKER_AUTH_KEY, required=False, 
        help="API server worker auth key",
    )

    
    return parser.parse_args()

def get_parameter(parameter_name, parameter_type, default_value, args, job_data, local_rank):
    parameter = default_value
    if local_rank == 0:
        if getattr(args, parameter_name) is not None:
            parameter = getattr(args, parameter_name)
        elif parameter_type(job_data[parameter_name]) is not None:
            parameter = parameter_type(job_data[parameter_name]) 
    parameter_list = [parameter]
    torch.distributed.broadcast_object_list(parameter_list, 0)
    return parameter_list[0]



class ProcessOutputCallback():
    
    PROGRESS_UPDATES_PER_SEC = 5

    def __init__(self, local_rank, api_worker, model_name, max_seq_len):
        self.local_rank = local_rank
        self.api_worker = api_worker
        self.model_name = model_name
        self.max_seq_len = max_seq_len
        self.progress_update_data = {}
        self.last_progress_update = time.time()

    def process_output(self, batch_idx, output, num_generated_tokens, finished):
        if self.local_rank == 0:
            job_batch_data = self.api_worker.get_current_job_batch_data()
            job_data = job_batch_data[batch_idx]
            result = {'text': output, 'model_name': self.model_name, 'num_generated_tokens': num_generated_tokens, 'max_seq_len': self.max_seq_len}
            if finished:
                self.progress_update_data.pop(batch_idx, None)
                return self.api_worker.send_job_results(result, job_data=job_data)
            else:
                self.progress_update_data[batch_idx] = result
                now = time.time()
                if (now - self.last_progress_update) > (1.0 / ProcessOutputCallback.PROGRESS_UPDATES_PER_SEC):
                    self.last_progress_update = now
                    progress_values = []
                    results = []
                    progress_job_batch_data = []
                    for idx in self.progress_update_data.keys():
                        result = self.progress_update_data[idx]
                        results.append(result)
                        progress_values.append(result.get('num_generated_tokens', 0))
                        progress_job_batch_data.append(job_batch_data[idx])
                    self.progress_update_data = {}
                    return self.api_worker.send_batch_progress(progress_values, results, job_batch_data=progress_job_batch_data)


class ProcessOutputToShellCallback():
    def __init__(self, local_rank, ctx):
        self.local_rank = local_rank
        self.ctx = ctx
        self.current_answer = None

    def process_output(self, batch_idx, output, num_generated_tokens, finished):
        if self.local_rank == 0:
            token = output.replace(self.current_answer, '') if self.current_answer else output
            print(token, end='', flush=True)
            self.current_answer = output if not finished else None

        if finished:
            self.ctx.append(
                {
                    "role": "assistant", 
                    "content": output
                }
            )



if __name__ == "__main__":
    main()
