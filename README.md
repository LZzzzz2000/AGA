# AGA
Attention-Guided Jailbreak Attacks on Large Reasoning Model



## Installation

1. Download the weights for Qwen/Qwen3-8B, mlabonne/Qwen3-8B-abliterated, and meta-llama/Llama-3.1-70B.
2. Change the ```ATTACK_MODEL_NAME``` and ```TARGET_MODEL_NAME``` in ```main.py```, ```JUDGE_MODEL_NAME``` in ```judge.py```



## Get API Key

Please get ```openai_api_key``` from [OpenAI API](https://platform.openai.com/docs/overview).

```python
client = OpenAI(
            base_url='openai',
            api_key='api_key' # add your api_key
        )
```



## Experiments

1. Change the ```DATA_PATH``` and ```SAVE_PATH``` in ```main.py```.
2. run ```python main.py```
