# Finetune Kimi-Audio

## 1. Data

We provide the demo data for each SFT task. You can prepare your own data in the same format.

The data file is a jsonl file, each line is a data in json format.


### Audio Understanding

The data format is as follows (we use ASR task as an example):

```json
{
    "task_type": "understanding",
    "conversation": [
        {
            "role": "user",
            "message_type": "text",
            "content": "Please transcribe the spoken content into written text."
        },
        {
            "role": "user",
            "message_type": "audio",
            "content": # Audio Path
        },
        {
            "role": "assistant",
            "message_type": "text",
            "content": # Transcript
        }
    ]
}
```

### Speech-to-Speech

For speech-to-speech tasks, the data format is as follows:

```json
{
    "task_type": "speech2speech",
    "conversation": [
        {
            "role": "user",
            "message_type": "audio",
            "content": # Input Audio Path
        },
        {
            "role": "assistant",
            "message_type": "audio",
            "content": # Output Audio Path
        }
    ]
}
```

* Librispeech ASR task as an example:
``` bash
python finetune_codes/demo_data/audio_understanding/prepare_librispeech_asrtask.py --output_dir "output/data/librispeech"
```

Note: The file `finetune_codes/demo_data/audio_understanding/data.jsonl` is the demo data for Librispeech ASR task, and it does not contain enough data for finetune. You can prepare your own data in the same format (or run the script `prepare_librispeech_asrtask.py` to generate the data) and put it in the `finetune_codes/demo_data/audio_understanding/` directory.

## 2. Finetune

1. Download the pretrained model and save it in `output/pretrained_hf`.

``` bash
CUDA_VISIBLE_DEVICES=0 python -m finetune_codes.model --model_name "moonshotai/Kimi-Audio-7B" --output_dir "output/pretrained_hf"
```

2. Preprocess the data and extract the semantic tokens.
```bash
CUDA_VISIBLE_DEVICES=0 python -m finetune_codes.extract_semantic_codes --input_file "finetune_codes/demo_data/audio_understanding/data.jsonl" --output_file "finetune_codes/demo_data/audio_understanding/data_with_semantic_codes.jsonl"
```

3. Finetune the model.

You can use the following command to finetune the model.

```bash
bash finetune_codes/finetune_ds.sh \
    --model_path "output/pretrained_hf" \
    --data "finetune_codes/demo_data/audio_understanding/data_with_semantic_codes.jsonl"
```

4. Convert the finetuned model for inference.
```bash
CUDA_VISIBLE_DEVICES=0 python -m finetune_codes.model --model_name "moonshotai/Kimi-Audio-7B" \
--action "export_model" \
--input_dir "output/kimiaudio_ckpts" \
--output_dir "output/finetuned_hf_for_inference"
```

You can infer with the finetuned model by running:
```bash
CUDA_VISIBLE_DEVICES=0 python -m finetune_codes.check_sft_infer
```

# Note

In this example, we support the ASR task and speech-to-speech task. For other tasks such as speech conversation or text-to-speech with text prompts, you might need to change the `tokenize_message` function in `finetune_codes/datasets.py`.

For speech-to-speech tasks, make sure to:
1. Prepare your data in the format shown above with `task_type: "speech2speech"`
2. Ensure both input and output audio files are accessible at the specified paths
3. Run the semantic code extraction on both input and output audio files

The hyper-parameters in `finetune_codes/finetune_ds.sh` should be tuned in new task because of the differences in task and dataset size.