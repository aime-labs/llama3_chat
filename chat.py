# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import List, Optional

from llama import Llama, Dialog

import argparse
from pathlib import Path
import os
import time
import torch
import random
import numpy as np

WORKER_JOB_TYPE = "llama2"
WORKER_AUTH_KEY = "66745b07e305b50505ca2b3284b4ae5f65d1"
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
        args.tokenizer_path = str(Path(args.ckpt_dir).parent / 'tokenizer.model')
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if args.api_server:

        from aime_api_worker_interface import APIWorkerInterface
        api_worker = APIWorkerInterface(args.api_server, WORKER_JOB_TYPE, WORKER_AUTH_KEY, args.gpu_id, world_size=world_size, rank=local_rank, gpu_name=torch.cuda.get_device_name(), worker_version=VERSION)
        callback = ProcessOutputCallback(local_rank, api_worker, Path(args.ckpt_dir).name)


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
                    ctx = job_data['text']
                    prompts.append(ctx)
                top_ps = api_worker.get_job_batch_parameter('top_p')
                top_ks = api_worker.get_job_batch_parameter('top_k')
                temperatures = api_worker.get_job_batch_parameter('temperature')
            else:
                prompts = [''] * batch_size # array has to be same size for multi rank broadcast
                top_ps = [args.top_p] * batch_size # array has to be same size for multi rank broadcast
                top_ks = [args.top_k] * batch_size # array has to be same size for multi rank broadcast
                temperatures = [args.temperature] * batch_size # array has to be same size for multi rank broadcast

            # synchronize across ranks
            torch.distributed.broadcast_object_list(prompts, 0)
            torch.distributed.broadcast_object_list(top_ps, 0)
            torch.distributed.broadcast_object_list(top_ks, 0)
            torch.distributed.broadcast_object_list(temperatures, 0)

            results = generator.generate_realtime(
                callback, prompts, max_gen_len=1024, temperatures=temperatures, top_ps=top_ps, top_ks=top_ks, repetition_penalty=args.repetition_penalty
            )

            print('Done')
    else:
        
        ctx = "A dialog, where User interacts with an helpful, kind, obedient, honest and very reasonable assistant called Dave.\n" +\
              "User: Hello, Dave.\n" +\
              "Dave: How can I assist you today?\n"

        callback = ProcessOutputToShellCallback(local_rank, ctx)
        print(f'\n{ctx}', end='', flush=True)
        while True:
            if local_rank == 0:
                prompt = input(f'User: ')
                if ctx != "":                    
                    print("Dave: ", end='', flush=True)
                    ctx = ctx + "User: " + prompt + "\n" + "Dave: "
                else:
                    ctx = prompt + "\n"
                
                prompts = [ctx]
            else:
                prompts = ['']
            torch.distributed.broadcast_object_list(prompts, src=0)
            if not args.temperature:
                args.temperature = 0.8
            if not args.top_p:
                args.top_p = 0.9
            if not args.top_k:
                args.top_k = 40
            results = generator.generate_realtime(
                callback, prompts, max_gen_len=1024, temperatures=[args.temperature], top_ps=[args.top_p], top_ks=[args.top_k], repetition_penalty=args.repetition_penalty
            )

            ctx = callback.ctx


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
        "--max_seq_len", type=int, default=4096, required=False,
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

    def __init__(self, local_rank, api_worker, model_name):
        self.local_rank = local_rank
        self.api_worker = api_worker
        self.model_name = model_name
        self.progress_update_data = {}
        self.last_progress_update = time.time()

    def process_output(self, batch_idx, output, num_generated_tokens, finished):
        if self.local_rank == 0:
            job_batch_data = self.api_worker.get_current_job_batch_data()
            job_data = job_batch_data[batch_idx]
            result = {'text': output, 'model_name': self.model_name, 'num_generated_tokens': num_generated_tokens}
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
        self.previous_token = None

    def process_output(self, batch_idx, output, num_generated_tokens, finished):
        if self.previous_token:
            token = output.split(self.previous_token)[-1]
        else:
            token = output

        print(token, end='', flush=True)
        
        if finished:
            self.ctx = output
            self.previous_token = None  
        else:
            if token:
                self.previous_token = token



if __name__ == "__main__":
    main()
