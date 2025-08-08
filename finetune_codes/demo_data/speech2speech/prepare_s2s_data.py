# finetune_codes/prepare_speech2speech_data.py
import argparse
import json
import os
import glob
import random
from tqdm import tqdm

def convert_single_file_data(data, path_prefix=""):
    """
    转换单条数据为speech2speech格式
    """
    # 构建绝对路径
    instruction_audio_path = os.path.join(path_prefix, data["instruction_wav_path"]) if path_prefix else data["instruction_wav_path"]
    response_audio_path = os.path.join(path_prefix, data["response_wav_path"]) if path_prefix else data["response_wav_path"]
    
    # 构建conversation格式
    conversation = [
        {
            "role": "user",
            "message_type": "audio",
            "content": instruction_audio_path
        },
        {
            "role": "assistant", 
            "message_type": "text",
            "content": data["response_text"]
        },
        {
            "role": "assistant",
            "message_type": "audio", 
            "content": response_audio_path
        }
    ]
    
    # 构建新的数据格式
    converted_data = {
        "task_type": "speech2speech",
        "conversation": conversation,
    }
    
    return converted_data

def convert_to_speech2speech_format(input_files, output_file, path_prefix="", shuffle=True):
    """
    将多个原始数据文件转换为speech2speech训练格式，支持合并和shuffle
    """
    all_converted_data = []
    total_processed = 0
    total_errors = 0
    
    # 处理每个输入文件
    for input_file in input_files:
        print(f"Processing file: {input_file}")
        file_processed = 0
        file_errors = 0
        
        with open(input_file, 'r', encoding='utf-8') as f_in:
            lines = f_in.readlines()
            
            for line_num, line in enumerate(tqdm(lines, desc=f"Converting {os.path.basename(input_file)}")):
                try:
                    data = json.loads(line.strip())
                    converted_data = convert_single_file_data(data, path_prefix)
                    all_converted_data.append(converted_data)
                    file_processed += 1
                    
                except Exception as e:
                    print(f"Error processing line {line_num + 1} in {input_file}: {e}")
                    file_errors += 1
                    continue
        
        print(f"  ✓ Processed: {file_processed} samples, Errors: {file_errors}")
        total_processed += file_processed
        total_errors += file_errors
    
    # Shuffle数据
    if shuffle:
        print("Shuffling data...")
        random.shuffle(all_converted_data)
    
    # 写入输出文件
    print(f"Writing to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for data in tqdm(all_converted_data, desc="Writing data"):
            f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    print("\n📊 Dataset Statistics:")
    print(f"  Total input files: {len(input_files)}")
    print(f"  Total samples processed: {total_processed}")
    print(f"  Total errors: {total_errors}")
    print(f"  Final dataset size: {len(all_converted_data)}")
    print(f"  Output file: {output_file}")
    
    return len(all_converted_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert UltraVoice data to Kimi-Audio speech2speech format")
    parser.add_argument("--input_dir", type=str, default="/home/tuwenming/Projects/UltraVoice_dev/data/metadata_tiny",
                       help="输入目录路径，包含所有jsonl文件")
    parser.add_argument("--input_pattern", type=str, default="ultravoice_*.jsonl",
                       help="输入文件匹配模式")
    parser.add_argument("--output_dir", type=str, default="/home/tuwenming/Projects/Kimi-Audio/finetune_codes/demo_data/speech2speech",
                       help="输出目录路径")
    parser.add_argument("--output_file", type=str, default="ultravoice_kimiaudio_sft.jsonl",
                       help="输出文件名")
    parser.add_argument("--path_prefix", type=str, default="/share/nlp/tuwenming/projects/UltraVoice_dev/data",
                       help="音频文件路径前缀，用于构建绝对路径")
    parser.add_argument("--no_shuffle", action="store_true",
                       help="禁用数据shuffle")
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子")
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    
    # 查找所有输入文件
    input_pattern = os.path.join(args.input_dir, args.input_pattern)
    input_files = glob.glob(input_pattern)
    input_files.sort()  # 确保处理顺序一致
    
    if not input_files:
        print(f"❌ No files found matching pattern: {input_pattern}")
        exit(1)
    
    print(f"Found {len(input_files)} files to process:")
    for f in input_files:
        print(f"  - {os.path.basename(f)}")
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 构建完整输出路径
    output_path = os.path.join(args.output_dir, args.output_file)
    
    print("\n🚀 Starting conversion...")
    print(f"Input directory: {args.input_dir}")
    print(f"Output file: {output_path}")
    if args.path_prefix:
        print(f"Path prefix: {args.path_prefix}")
    print(f"Shuffle: {not args.no_shuffle}")
    
    # 执行转换
    total_samples = convert_to_speech2speech_format(
        input_files=input_files,
        output_file=output_path,
        path_prefix=args.path_prefix,
        shuffle=not args.no_shuffle
    )
    
    
    print("\n✅ Conversion completed!")
    print(f"📁 Output saved to: {output_path}")
    print(f"📊 Total samples: {total_samples}")