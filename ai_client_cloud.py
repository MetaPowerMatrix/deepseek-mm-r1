#!/usr/bin/env python3
import asyncio
import json
import logging
import os
import sys
import datetime
import wave
import uuid
import argparse
import websockets
from vosk import Model, KaldiRecognizer
import requests
from pydub import AudioSegment
from dotenv import load_dotenv
from tetos.volc import VolcSpeaker

# 加载.env文件中的环境变量
load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("ai_client")

# 全局变量
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
AUDIO_DIR = os.getenv("AUDIO_DIR", "audio_files")
PROCESSED_DIR = os.getenv("PROCESSED_DIR", "processed_files")
TTS_VOICE = os.getenv("TTS_VOICE", "zh-CN-XiaoxiaoNeural")
WS_URL = os.getenv("WS_URL", "ws://stream.kalaisai.com:80/ws/proxy")
VOSK_MODEL_PATH = os.getenv("VOSK_MODEL_PATH", "vosk-model-cn-0.22")
VOLC_ACCESS_KEY = os.getenv("VOLC_ACCESS_KEY", "")
VOLC_SECRET_KEY = os.getenv("VOLC_SECRET_KEY", "")
VOLC_APP_KEY = os.getenv("VOLC_APP_KEY", "")

# 初始化Tetos EdgeSpeaker
volc_speaker = VolcSpeaker(access_key=VOLC_ACCESS_KEY, secret_key=VOLC_SECRET_KEY, app_key=VOLC_APP_KEY)

# 全局Vosk模型实例
vosk_model = None

# 音频参数 - ESP32兼容
ESP32_SAMPLE_RATE = 16000
ESP32_CHANNELS = 1
ESP32_SAMPLE_WIDTH = 2  # 16位

# 会话历史记录
conversation_history = []

def setup_directories():
    """确保必要的目录存在"""
    os.makedirs(AUDIO_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    logger.info(f"已创建目录: {AUDIO_DIR}, {PROCESSED_DIR}")

def initialize_vosk_model():
    """初始化Vosk模型(全局单例)"""
    global vosk_model
    
    if vosk_model is not None:
        return vosk_model
    
    # 限制内存使用
    os.environ['VOSK_DEBUG'] = '0'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    
    try:
        if not os.path.exists(VOSK_MODEL_PATH):
            logger.error(f"Vosk模型目录不存在: {VOSK_MODEL_PATH}")
            return None
        
        logger.info(f"正在加载Vosk模型: {VOSK_MODEL_PATH}")
        vosk_model = Model(VOSK_MODEL_PATH)
        logger.info("Vosk模型加载成功")
        return vosk_model
    except Exception as e:
        logger.error(f"初始化Vosk模型失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def vosk_speech_to_text(audio_path):
    """
    使用Vosk离线识别（使用全局模型实例）
    """
    global vosk_model
    
    try:
        # 确保模型已初始化
        if vosk_model is None:
            vosk_model = initialize_vosk_model()
            if vosk_model is None:
                return "语音识别失败：模型未找到或无法加载"
        
        with wave.open(audio_path, 'rb') as wf:
            sample_rate = wf.getframerate()
            # 检查采样率是否在合理范围内
            if sample_rate < 8000 or sample_rate > 48000:
                logger.warning(f"异常采样率: {sample_rate}, 使用默认16000")
                sample_rate = 16000
                
            recognizer = KaldiRecognizer(vosk_model, sample_rate)
            
            text = []
            while True:
                data = wf.readframes(4000)  # 每次读取4000帧进行处理
                if len(data) == 0:
                    break
                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    text.append(result.get("text", ""))
            
            final_result = json.loads(recognizer.FinalResult())
            text.append(final_result.get("text", ""))
            
        return " ".join([t for t in text if t])  # 过滤空字符串
    except Exception as e:
        logger.error(f"语音识别过程出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return "语音识别失败：处理过程出错"

async def save_raw_to_wav(raw_data, wav_file_path):
    """将原始PCM数据保存为WAV文件"""
    with wave.open(wav_file_path, 'wb') as wav_file:
        wav_file.setnchannels(ESP32_CHANNELS)
        wav_file.setsampwidth(ESP32_SAMPLE_WIDTH)
        wav_file.setframerate(ESP32_SAMPLE_RATE)
        wav_file.writeframes(raw_data)
    return wav_file_path

import re

def remove_markdown(text):
    # 去除加粗、斜体等标记
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # **加粗** → 加粗
    text = re.sub(r'\*(.*?)\*', r'\1', text)      # *斜体* → 斜体
    text = re.sub(r'`(.*?)`', r'\1', text)        # `代码` → 代码
    text = re.sub(r'#+\s*', '', text)             # 去除标题（### 标题 → 标题）
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)  # 超链接[文字](url) → 文字
    return text.strip()

async def get_deepseek_response(prompt):
    """调用DeepSeek API获取回复"""
    try:
        # 保存对话历史以维持上下文
        global conversation_history
        
        # 检查API密钥
        if not DEEPSEEK_API_KEY:
            logger.error("环境变量DEEPSEEK_API_KEY未设置")
            return "抱歉，AI服务未配置正确。"
        
        # 构建完整消息历史
        prompt = prompt + " 请直接输出纯文本，不要使用Markdown格式（如不要加粗、标题、代码块等）。"
        messages = conversation_history + [{"role": "user", "content": prompt}]
        
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "deepseek-chat",
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 800,
        }
        
        logger.info("正在调用DeepSeek API...")
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            json=data
        )
        
        if response.status_code == 200:
            result = response.json()
            assistant_response = result["choices"][0]["message"]["content"]
            
            # 更新对话历史
            conversation_history.append({"role": "user", "content": prompt})
            conversation_history.append({"role": "assistant", "content": assistant_response})
            
            # 保持对话历史在合理长度
            if len(conversation_history) > 10:
                conversation_history = conversation_history[-10:]
                
            logger.info(f"DeepSeek回复: {assistant_response}")
            return assistant_response
        else:
            logger.error(f"DeepSeek API错误: {response.status_code}")
            logger.error(response.text)
            return "抱歉，我无法获取回复。请稍后再试。"
    except Exception as e:
        logger.error(f"调用DeepSeek API错误: {e}")
        return "抱歉，处理您的请求时出现了错误。"


async def text_to_speech(text):
    """文本转语音，使用Tetos的EdgeSpeaker"""
    temp_file = None
    
    try:
        # 生成临时文件路径
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        temp_path = os.path.join(PROCESSED_DIR, f"temp_tts_{timestamp}.mp3")
        temp_file = temp_path
        logger.info(f"mp3生成临时文件路径: {temp_path}")
        
        # 使用Tetos的EdgeSpeaker生成语音
        global volc_speaker
        length = await volc_speaker.synthesize(text, temp_path, lang="zh-CN")
        logger.info(f"Tetos生成语音文件完成: {temp_path}, 长度: {length}")
        
        # 用pydub处理
        audio = AudioSegment.from_file(temp_path, format="mp3")
        audio = audio.set_frame_rate(ESP32_SAMPLE_RATE)
        audio = audio.set_channels(ESP32_CHANNELS)
        audio = audio.set_sample_width(ESP32_SAMPLE_WIDTH)
        
        # 导出为PCM
        pcm_data = audio.raw_data
        logger.info(f"生成PCM数据: {len(pcm_data)} 字节")

        # 调试：验证数据有效性
        assert len(pcm_data) % (ESP32_SAMPLE_WIDTH * ESP32_CHANNELS) == 0, "无效的PCM数据长度"
        
        return pcm_data
        
    except Exception as e:
        logger.error(f"TTS错误: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None
    finally:
        # 删除临时MP3文件
        try:
            if temp_file and os.path.exists(temp_file):
                os.remove(temp_file)
                logger.debug(f"已删除临时TTS文件: {temp_file}")
        except Exception as e:
            logger.warning(f"删除临时TTS文件失败: {e}")

            
async def process_audio(raw_audio_data, session_id):
    """处理音频数据的完整流程"""
    temp_files = []  # 记录临时文件以便清理
    
    try:
        # 生成唯一文件名
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        wav_file_path = os.path.join(AUDIO_DIR, f"audio_{session_id}_{timestamp}.wav")
        temp_files.append(wav_file_path)
        
        # 将原始数据保存为WAV文件
        await save_raw_to_wav(raw_audio_data, wav_file_path)
        logger.info(f"已保存WAV文件: {wav_file_path}")
        
        # 转录音频
        logger.info("开始语音识别...")
        
        transcript = vosk_speech_to_text(wav_file_path)
        if not transcript:
            logger.warning("语音识别失败，未能获取文本")
            return None, "抱歉，无法识别您的语音。"
        
        logger.info(f"语音识别结果: {transcript}")
        
        # 获取DeepSeek回复
        logger.info("正在获取AI回复...")
        ai_response = await get_deepseek_response(transcript)
        # 清理markdown格式
        ai_response = remove_markdown(ai_response)
        logger.info(f"AI回复(已清理格式): {ai_response}")

        # 生成语音回复
        logger.info("正在生成语音回复...")
        audio_response = await text_to_speech(ai_response)

        # 如果成功生成语音
        if audio_response:
            logger.info(f"已生成语音回复: {len(audio_response)} 字节")
            return audio_response, ai_response
        
        logger.warning("语音合成失败")
        return None, ai_response
    except Exception as e:
        logger.error(f"处理音频流程出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, "处理请求时发生错误。"
    finally:
        # 清理临时文件
        for file_path in temp_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.debug(f"已清理临时文件: {file_path}")
            except Exception as e:
                logger.warning(f"清理临时文件失败: {file_path}, 错误: {e}")

async def ai_backend_client(websocket_url):
    """
    AI后端客户端
    连接到WebSocket代理，接收音频数据，处理后返回结果
    """
    # 重连参数
    max_reconnect_attempts = 10
    reconnect_delay_seconds = 5
    reconnect_attempt = 0
    
    # 预初始化Vosk模型
    logger.info("预加载Vosk模型...")
    initialize_vosk_model()
    
    while reconnect_attempt <= max_reconnect_attempts:
        try:
            if reconnect_attempt > 0:
                logger.info(f"尝试重新连接 (尝试 {reconnect_attempt}/{max_reconnect_attempts})...")
            else:
                logger.info(f"正在连接到WebSocket代理: {websocket_url}")
            
            # 连接到WebSocket
            async with websockets.connect(websocket_url) as websocket:
                # 重置重连计数
                reconnect_attempt = 0
                
                # 发送AI后端标识
                await websocket.send(json.dumps({
                    "client_type": "ai_backend"
                }))
                
                # 接收连接确认
                response = await websocket.recv()
                data = json.loads(response)
                logger.info(f"连接确认: {data.get('content', '')}")
                
                # 处理请求循环
                while True:
                    try:
                        # 接收消息
                        message = await websocket.recv()
                        
                        # 判断消息类型 - 文本还是二进制
                        if isinstance(message, str):
                            # 文本消息 - 可能是控制命令
                            try:
                                data = json.loads(message)
                                
                                # 处理取消请求
                                if data.get("type") == "cancel_processing":
                                    session_id = data.get("session_id")
                                    logger.info(f"取消处理会话 {session_id}")
                                    # 这里可以添加取消正在进行的处理逻辑
                                
                            except json.JSONDecodeError:
                                logger.error(f"无法解析JSON消息: {message}")
                        
                        elif isinstance(message, bytes):
                            # 直接处理二进制数据（音频）
                            binary_data = message
                            
                            # 从二进制数据中提取会话ID（前16字节）和音频数据
                            if len(binary_data) > 16:
                                # 提取会话ID（UUID格式，存储在前16字节）
                                session_id_bytes = binary_data[:16]
                                raw_audio = binary_data[16:]
                                
                                try:
                                    # 将字节转换为UUID字符串
                                    session_id = uuid.UUID(bytes=session_id_bytes).hex
                                    logger.info(f"收到音频数据: 会话ID = {session_id}, 大小 = {len(raw_audio)} 字节")
                                    
                                    # 发送处理状态
                                    await websocket.send(json.dumps({
                                        "type": "text",
                                        "session_id": session_id,
                                        "content": "正在处理音频..."
                                    }))
                                    
                                    # 处理音频数据
                                    audio_response, text_response = await process_audio(raw_audio, session_id)
                                    
                                    # 发送文本回复
                                    await websocket.send(json.dumps({
                                        "type": "text",
                                        "session_id": session_id,
                                        "content": text_response
                                    }))
                                    
                                    # 发送音频回复 - 分块发送
                                    if audio_response:
                                        chunk_size = 5120  # 大约5KB
                                        for i in range(0, len(audio_response), chunk_size):
                                            # 截取一块音频数据
                                            audio_chunk = audio_response[i:i+chunk_size]
                                            # 添加会话ID前缀
                                            data_with_session = session_id_bytes + audio_chunk
                                            # 发送
                                            await websocket.send(data_with_session)
                                            logger.info(f"发送音频回复块: 会话ID = {session_id}, 块大小 = {len(audio_chunk)} 字节")
                                            # 短暂暂停，避免发送过快
                                            await asyncio.sleep(0.05)
                                    
                                    logger.info(f"会话 {session_id} 处理完成")
                                    
                                except ValueError:
                                    logger.error("无法解析会话ID")
                            else:
                                logger.error(f"收到无效的音频数据: 长度过短 ({len(binary_data)} 字节)")
                    
                    except Exception as e:
                        import sys
                        exc_type, exc_obj, exc_tb = sys.exc_info()
                        line_number = exc_tb.tb_lineno
                        logger.error(f"处理消息时出错: {str(e)}, 出错行号: {line_number}")
                        
        except websockets.exceptions.ConnectionClosed as e:
            reconnect_attempt += 1
            logger.warning(f"WebSocket连接已关闭: {str(e)}")
            if reconnect_attempt <= max_reconnect_attempts:
                logger.info(f"将在 {reconnect_delay_seconds} 秒后尝试重新连接...")
                await asyncio.sleep(reconnect_delay_seconds)
                # 指数退避算法增加重连延迟
                reconnect_delay_seconds = min(60, reconnect_delay_seconds * 1.5)
            else:
                logger.error(f"达到最大重连次数 ({max_reconnect_attempts})，停止重连")
                break
                
        except Exception as e:
            reconnect_attempt += 1
            import sys
            exc_type, exc_obj, exc_tb = sys.exc_info()
            line_number = exc_tb.tb_lineno
            logger.error(f"WebSocket连接错误: {str(e)}, 出错行号: {line_number}")
            
            if reconnect_attempt <= max_reconnect_attempts:
                logger.info(f"将在 {reconnect_delay_seconds} 秒后尝试重新连接...")
                # 收集垃圾
                import gc
                gc.collect()
                
                await asyncio.sleep(reconnect_delay_seconds)
                # 指数退避算法增加重连延迟
                reconnect_delay_seconds = min(60, reconnect_delay_seconds * 1.5)
            else:
                logger.error(f"达到最大重连次数 ({max_reconnect_attempts})，停止重连")
                break
    
    logger.error("WebSocket客户端已停止运行")

def main():
    # 参数解析
    parser = argparse.ArgumentParser(description="AI音频处理客户端")
    parser.add_argument("--url", type=str, default="ws://localhost:8000/api/proxy", 
                      help="WebSocket代理URL")
    parser.add_argument("--model", type=str, default="vosk-model-cn-0.22", 
                      help="Vosk语音识别模型路径")
    parser.add_argument("--audio-dir", type=str, default="audio_files",
                      help="音频文件存储目录")
    parser.add_argument("--processed-dir", type=str, default="processed_files",
                      help="处理后文件存储目录")
    parser.add_argument("--api-key", type=str, 
                      help="DeepSeek API密钥")
    
    args = parser.parse_args()
    
    # 更新全局变量
    global DEEPSEEK_API_KEY, AUDIO_DIR, PROCESSED_DIR
    if args.api_key:
        DEEPSEEK_API_KEY = args.api_key

    # 创建必要的目录
    setup_directories()
    
    # 检查API密钥
    if not DEEPSEEK_API_KEY:
        logger.warning("未设置DeepSeek API密钥，将无法使用DeepSeek服务")
    
    # 打印启动信息
    logger.info("=" * 50)
    logger.info("AI音频处理客户端启动")
    logger.info(f"WebSocket URL: {WS_URL}")
    logger.info(f"Vosk模型路径: {args.model}")
    logger.info(f"音频文件目录: {AUDIO_DIR}")
    logger.info(f"处理文件目录: {PROCESSED_DIR}")
    logger.info("=" * 50)
    
    # 启动异步循环
    asyncio.run(ai_backend_client(WS_URL))

if __name__ == "__main__":
    main()# AI音频处理客户端程序
