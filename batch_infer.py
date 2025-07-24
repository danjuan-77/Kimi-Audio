from kimia_infer.api.kimia import KimiAudio
import os
import soundfile as sf
import argparse
import json
import sys
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="批量音频推理脚本")
    parser.add_argument("--model_path", type=str, default="/share/nlp/tuwenming/models/moonshotai/Kimi-Audio-7B-Instruct", 
                       help="模型路径")
    parser.add_argument("--test_jsonl", type=str, default="/share/nlp/tuwenming/projects/UltraVoice_dev/eval/ultravoice_testset.jsonl",
                       help="测试数据jsonl文件路径")
    parser.add_argument("--wav_prefix", type=str, default="/share/nlp/tuwenming/projects/UltraVoice_dev/data",
                       help="音频文件路径前缀")
    parser.add_argument("--save_dir", type=str, default="infer_results/kimiaudio-7b-instruct/wav",
                       help="输出音频保存目录")
    
    args = parser.parse_args()
    
    print(f"正在加载模型: {args.model_path}")
    
    # 初始化模型
    model = KimiAudio(
        model_path=args.model_path,
        load_detokenizer=True,
    )
    
    # 设置采样参数
    sampling_params = {
        "audio_temperature": 0.8,
        "audio_top_k": 10,
        "text_temperature": 0.0,
        "text_top_k": 5,
        "audio_repetition_penalty": 1.0,
        "audio_repetition_window_size": 64,
        "text_repetition_penalty": 1.0,
        "text_repetition_window_size": 16,
    }
    
    # 创建输出目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    print(f"开始批处理推理，输入文件: {args.test_jsonl}")
    
    processed_count = 0
    error_count = 0
    
    try:
        # 先计算总行数
        with open(args.test_jsonl, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)
        
        # 进行批处理
        with open(args.test_jsonl, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(tqdm(f, total=total_lines, desc="批处理推理"), 1):
                try:
                    data = json.loads(line.strip())
                    
                    # 提取instruction_wav_path
                    if 'instruction_wav_path' not in data:
                        print(f"警告: 第{line_num}行缺少instruction_wav_path字段")
                        error_count += 1
                        continue
                    
                    wav_path = data['instruction_wav_path']
                    
                    # 添加前缀路径
                    full_wav_path = os.path.join(args.wav_prefix, wav_path)
                    
                    # 检查文件是否存在
                    if not os.path.exists(full_wav_path):
                        print(f"警告: 音频文件不存在: {full_wav_path}")
                        error_count += 1
                        continue
                    
                    # 构建输出文件名
                    try:
                        output_filename = f"{data['split_type']}_{data['sub_type']}_{data['data_source']}_{data['index']}.wav"
                    except KeyError as e:
                        print(f"警告: 第{line_num}行缺少必要字段 {e}，使用默认文件名")
                        base_name = os.path.basename(full_wav_path)
                        output_filename = f"{base_name.replace('.wav', '')}_output.wav"
                        error_count += 1
                        continue
                    
                    # 检查输出文件是否已存在
                    output_path = os.path.join(args.save_dir, output_filename)
                    if os.path.exists(output_path):
                        print(f"跳过第{line_num}行: 输出文件已存在: {output_path}")
                        processed_count += 1
                        continue
                    
                    print(f"正在处理第{line_num}行: {full_wav_path}")
                    
                    # 构建消息 - audio2audio模式
                    messages = [
                        {
                            "role": "user",
                            "message_type": "audio",
                            "content": full_wav_path,
                        }
                    ]
                    
                    # 进行推理
                    try:
                        wav, text = model.generate(messages, **sampling_params, output_type="both")
                        
                        # 保存音频文件
                        sf.write(
                            output_path,
                            wav.detach().cpu().view(-1).numpy(),
                            24000,
                        )
                        
                        processed_count += 1
                        
                        print(f"完成第{line_num}行处理")
                        print(f"文本结果: {text}")
                        print(f"音频保存至: {output_path}")
                        
                    except Exception as e:
                        print(f"错误: 处理第{line_num}行时模型推理出错: {str(e)}")
                        error_count += 1
                        continue
                
                except json.JSONDecodeError as e:
                    print(f"错误: 第{line_num}行JSON解析失败: {str(e)}")
                    error_count += 1
                    continue
                
                except Exception as e:
                    print(f"错误: 处理第{line_num}行时出现未知错误: {str(e)}")
                    error_count += 1
                    continue
    
    except FileNotFoundError:
        print(f"错误: 找不到输入文件: {args.test_jsonl}")
        sys.exit(1)
    
    print(f"\n批处理完成!")
    print(f"总计处理: {processed_count} 条")
    print(f"错误数量: {error_count} 条")
    print(f"音频文件保存目录: {args.save_dir}")

if __name__ == "__main__":
    main() 