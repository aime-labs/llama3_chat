<p align="center">
  <img src="https://github.com/meta-llama/llama3/blob/main/Llama3_Repo.jpeg" width="400"/>
</p>

<p align="center">
        🤗 <a href="https://api.aime.info"> AIME API Server</a>&nbsp | <a href="https://ai.meta.com/blog/"> Blog</a>&nbsp |  <a href="https://llama.meta.com/">Website</a>&nbsp | <a href="https://llama.meta.com/get-started/">Get Started</a>&nbsp
<br>

---


# Meta Llama 3 / 3.1

We are unlocking the power of large language models. Our latest version of Llama is now accessible to individuals, creators, researchers, and businesses of all sizes so that they can experiment, innovate, and scale their ideas responsibly.

This release includes model weights and starting code for pre-trained and instruction tuned Llama 3 language models — including sizes of 8B to 70B parameters.

This repository is intended to run LLama 3 models as worker for the [AIME API Server](https://github.com/aime-team/aime-api-server) also an interactive console chat for testing purpose is available.

Llama 3 API demo server running at: [https://api.aime.info/llama3-chat/](https://api.aime.info/llama3-chat/)

## Features

* Realtime interactive console chat example
* Llama 3 70B support for 2 GPU (e.g. 2x A100/H100 80 GB) and 4 GPU (e.g. 4x A100 40GB/RTX A6000/6000 Ada) setups
* Worker mode for AIME API server to use Llama3 as HTTP/HTTPS API endpoint
* Batch job aggreation support for AIME API server for higher GPU throughput with multi-user chat

## Download

In order to download the model weights and tokenizer, please visit the [Meta Llama website](https://llama.meta.com/llama-downloads/) and accept our License.

Once your request is approved, you will receive a signed URL over email. Then run the download.sh script, passing the URL provided when prompted to start the download.

Pre-requisites: Make sure you have `wget` and `md5sum` installed. Then run the script: `./download.sh`.

Keep in mind that the links expire after 24 hours and a certain amount of downloads. If you start seeing errors such as `403: Forbidden`, you can always re-request a link.


## Quick Start

You can follow the steps below to quickly get up and running with Llama 3 models. These steps will let you run quick inference locally. For more examples, see the [Llama recipes repository](https://github.com/facebookresearch/llama-recipes).

1. In a conda env with PyTorch / CUDA available clone and download this repository.

2. In the top-level directory run:
    ```bash
    pip install -e .
    ```
3. Visit the [Meta Llama website](https://llama.meta.com/llama-downloads/) and register to download the model/s.

4. Once registered, you will get an email with a URL to download the models. You will need this URL when you run the download.sh script.

5. Once you get the email, navigate to your downloaded llama repository and run the download.sh script.
    - Make sure to grant execution permissions to the download.sh script
    - During this process, you will be prompted to enter the URL from the email.
    - Do not use the “Copy Link” option but rather make sure to manually copy the link from the email.

#### 6. Convert 70B models for 2 or 4 GPU configuration (if required)

The default sharding configuration of the downloaded Llama 3 70B model weights is for 8 GPUs (with 24 GB memory). The weights for a 4 or 2 GPU setups can be converted with the 'convert_weights.py' script.

To do so run following command:

```
python convert_weights.py --input_dir /data/models/Meta-Llama-3-70B-Instruct/ --model_size 70B --num_gpus <num_gpus>
```
<num_gpus> can be:

- 4 for 4x at least 40 GB memory per GPU
- 2 for 2x at least 80 GB memory per GPU

#### 7a. Start a Chat with LLama 3 / 3.1 in Command Line

Run the chat mode in the command line with following command:
```
torchrun --nproc_per_node <num_gpus> chat.py --ckpt_dir <destination_of_checkpoints>
```
It will start a single user chat (batch_size is 1) with Dave.

#### 7b. Start Llama 3 / 3.1 Chat as AIME API Worker

To run Llama 3 / 3.1 Chat as HTTP/HTTPS API with [AIME API Server](https://github.com/aime-team/aime-api-server) start the chat command with following command line:

```
torchrun --nproc_per_node <num_gpus> chat.py --ckpt_dir <destination_of_checkpoints> --api_server <url to api server>
```
It will start Llama 3 / 3.1 as a worker, waiting for job request through the AIME API Server. Use the --max_batch_size option to control how many parallel job requests can be handled (depending on the available GPU memory). 

**Note**
- Replace  `Meta-Llama-3-8B-Instruct/` with the path to your checkpoint directory and `Meta-Llama-3-8B-Instruct/tokenizer.model` with the path to your tokenizer model.
- The `–nproc_per_node` should be set to the [MP](#inference) value for the model you are using.
- Adjust the `max_seq_len` and `max_batch_size` parameters as needed.


## Inference

Different models require different model-parallel (MP) values:

|  Model | MP |
|--------|----|
| 8B     | 1  |
| 70B    | 8, 4, 2 (for 4 and 2 the weights have to be converted with the convert_weights script)  |

All models support sequence length up to 8192 tokens, but we pre-allocate the cache according to `max_seq_len` and `max_batch_size` values. So set those according to your hardware.

### Pretrained Models

These models are not finetuned for chat or Q&A. They should be prompted so that the expected answer is the natural continuation of the prompt.

See `example_text_completion.py` for some examples. To illustrate, see the command below to run it with the llama-3-8b model (`nproc_per_node` needs to be set to the `MP` value):

```
torchrun --nproc_per_node 1 example_text_completion.py \
    --ckpt_dir Meta-Llama-3-8B/ \
    --tokenizer_path Meta-Llama-3-8B/tokenizer.model \
    --max_seq_len 128 --max_batch_size 4
```

### Instruction-tuned Models

The fine-tuned models were trained for dialogue applications. To get the expected features and performance for them, a specific formatting defined in [`ChatFormat`](https://github.com/meta-llama/llama3/blob/main/llama/tokenizer.py#L202)
needs to be followed: The prompt begins with a `<|begin_of_text|>` special token, after which one or more messages follow. Each message starts with the `<|start_header_id|>` tag, the role `system`, `user` or `assistant`, and the `<|end_header_id|>` tag. After a double newline `\n\n` the contents of the message follow. The end of each message is marked by the `<|eot_id|>` token.

You can also deploy additional classifiers for filtering out inputs and outputs that are deemed unsafe. See the llama-recipes repo for [an example](https://github.com/meta-llama/llama-recipes/blob/main/recipes/inference/local_inference/inference.py) of how to add a safety checker to the inputs and outputs of your inference code.

Examples using llama-3-8b-chat:

```
torchrun --nproc_per_node 1 example_chat_completion.py \
    --ckpt_dir Meta-Llama-3-8B-Instruct/ \
    --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model \
    --max_seq_len 512 --max_batch_size 6
```

Llama 3 is a new technology that carries potential risks with use. Testing conducted to date has not — and could not — cover all scenarios.
In order to help developers address these risks, we have created the [Responsible Use Guide](https://ai.meta.com/static-resource/responsible-use-guide/).

## Issues

Please report any software “bug”, or other problems with the models through one of the following means:
- Reporting issues with the model: [https://github.com/meta-llama/llama3/issues](https://github.com/meta-llama/llama3/issues)
- Reporting risky content generated by the model: [developers.facebook.com/llama_output_feedback](http://developers.facebook.com/llama_output_feedback)
- Reporting bugs and security concerns: [facebook.com/whitehat/info](http://facebook.com/whitehat/info)

## Model Card
See [MODEL_CARD.md](MODEL_CARD.md).

## License

Our model and weights are licensed for both researchers and commercial entities, upholding the principles of openness. Our mission is to empower individuals, and industry through this opportunity, while fostering an environment of discovery and ethical AI advancements.

See the [LICENSE](LICENSE) file, as well as our accompanying [Acceptable Use Policy](USE_POLICY.md)

## Questions

For common questions, the FAQ can be found [here](https://llama.meta.com/faq) which will be kept up to date over time as new questions arise.
