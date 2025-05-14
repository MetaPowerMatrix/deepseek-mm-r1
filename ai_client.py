import asyncio
import json
import logging
import os
import datetime
import wave
import uuid
import argparse
import requests
from dotenv import load_dotenv
from pathlib import Path
from pydub import AudioSegment
import random
import websocket
import time

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
AUDIO_DIR = os.getenv("AUDIO_DIR", "audio_files")
PROCESSED_DIR = os.getenv("PROCESSED_DIR", "processed_files")
WS_URL = os.getenv("WS_URL", "ws://stream.kalaisai.com:80/ws/proxy")

# 本地服务接口URL
API_URL = "http://127.0.0.1:8000/api/v1"
TTS_API_URL = "http://127.0.0.1:5000/process"
SPEECH_TO_TEXT_URL = f"{API_URL}/speech-to-text"
CHAT_URL = f"{API_URL}/chat"
QWEN_CHAT_URL = f"{API_URL}/chat/qwen"
MINICPM_URL = f"{API_URL}/voice-chat"
TEXT_TO_SPEECH_URL = f"{TTS_API_URL}"
UNCENSORED_CHAT_URL = f"{API_URL}/chat/uncensored"

# 状态接口URL
SPEECH_TO_TEXT_STATUS_URL = f"{API_URL}/speech-to-text/status"
CHAT_STATUS_URL = f"{API_URL}/chat/status"
MEGATTS_STATUS_URL = f"{API_URL}/megatts/status"
MINICPM_STATUS_URL = f"{API_URL}/minicpm/status"
QWEN_CHAT_STATUS_URL = f"{API_URL}/qwen/status"
UNCENSORED_CHAT_STATUS_URL = f"{API_URL}/uncensored/status"

# 会话历史记录
conversation_history = []

# 全局变量
AUDIO_CATEGORIES = {}

# 全局配置变量
USE_MINICPM = False
USE_QWEN = False
SKIP_TTS = False
USE_F5TTS = False
USE_UNCENSORED = False

# WebSocket相关配置
WS_HEARTBEAT_INTERVAL = 30  # 心跳间隔（秒）
WS_RECONNECT_DELAY = 5  # 重连延迟（秒）
WS_MAX_RETRIES = 3  # 最大重试次数
WS_CHUNK_SIZE = 4096  # 数据分块大小
WS_SEND_TIMEOUT = 10  # 发送超时时间（秒）


def setup_directories():
    """确保必要的目录存在"""
    os.makedirs(AUDIO_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    logger.info(f"已创建目录: {AUDIO_DIR}, {PROCESSED_DIR}")

def check_service_status():
    """检查本地服务接口的状态"""
    try:
        # 检查MiniCPM服务状态
        if USE_MINICPM:
            response = requests.get(MINICPM_STATUS_URL)
            if response.status_code == 200:
                logger.info(f"MiniCPM服务状态: {response.json()}")
            else:
                logger.error(f"MiniCPM服务状态检查失败: {response.status_code}")

        if not USE_MINICPM:
            # 检查聊天服务状态
            if USE_QWEN:
                response = requests.get(QWEN_CHAT_STATUS_URL)
                if response.status_code == 200:
                    logger.info(f"Qwen聊天服务状态: {response.json()}")
                else:
                    logger.error(f"Qwen聊天服务状态检查失败: {response.status_code}")
            if USE_UNCENSORED:
                response = requests.get(UNCENSORED_CHAT_STATUS_URL)
                if response.status_code == 200:
                    logger.info(f"未审核聊天服务状态: {response.json()}")
                else:
                    logger.error(f"未审核聊天服务状态检查失败: {response.status_code}")
            else:
                response = requests.get(CHAT_STATUS_URL)
                if response.status_code == 200:
                    logger.info(f"Deepseek聊天服务状态: {response.json()}")
                else:
                    logger.error(f"Deepseek聊天服务状态检查失败: {response.status_code}")

            # 检查语音转文字服务状态
            response = requests.get(SPEECH_TO_TEXT_STATUS_URL)
            if response.status_code == 200:
                logger.info(f"语音转文字服务状态: {response.json()}")
            else:
                logger.error(f"语音转文字服务状态检查失败: {response.status_code}")

    except Exception as e:
        logger.error(f"服务状态检查失败: {e}")

def speech_to_text(audio_path):
    """调用本地服务接口将语音转换为文本"""
    try:
        logger.info(f"开始语音转文字请求: {audio_path}")
        
        # 检查文件是否存在
        if not os.path.exists(audio_path):
            logger.error(f"音频文件不存在: {audio_path}")
            return None
            
        # 记录文件大小
        file_size = os.path.getsize(audio_path)
        logger.info(f"音频文件大小: {file_size} 字节")
        
        with open(audio_path, 'rb') as audio_file:
            # 为文件指定名称、内容类型和文件对象
            files = {
                'file': (os.path.basename(audio_path), 
                         audio_file, 
                         'audio/wav')  # 指定 MIME 类型为 audio/wav
            }
            
            headers = {
                'Accept': 'application/json'
            }
            
            logger.info(f"发送请求到: {SPEECH_TO_TEXT_URL}")
            logger.info(f"请求头: {headers}")
            logger.info(f"文件名: {os.path.basename(audio_path)}")
            
            response = requests.post(SPEECH_TO_TEXT_URL, files=files, headers=headers)
            
            logger.info(f"收到响应: 状态码={response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                # 根据实际的返回格式解析
                if result.get("code") == 0:
                    transcription = result.get("data", {}).get("transcription", "")
                    language = result.get("data", {}).get("language", "")
                    logger.info(f"语音转文字成功，结果: {transcription}, 语言: {language}")
                    return transcription
                else:
                    logger.error(f"API返回错误: {result.get('message', '未知错误')}")
                    return None
            else:
                logger.error(f"语音转文字失败: 状态码={response.status_code}, 响应内容={response.text}")
                return None
    except Exception as e:
        logger.error(f"语音转文字接口调用失败: {str(e)}")
        import traceback
        logger.error(f"异常堆栈: {traceback.format_exc()}")
        return None

def get_chat_response(prompt):
    """调用聊天接口获取回复，根据配置选择Qwen或Deepseek"""
    global conversation_history
    
    # 确定要使用的URL
    if USE_UNCENSORED:
        url = UNCENSORED_CHAT_URL
    elif USE_QWEN:
        url = QWEN_CHAT_URL
    else:
        url = CHAT_URL

    model_name = "Qwen" if USE_QWEN else "Deepseek"
    model_name = "Uncensored" if USE_UNCENSORED else model_name

    try:
        data = {
            "prompt": prompt,
            "history": conversation_history,
            "max_length": 2048,
            "temperature": 0.7,
            "top_p": 0.9
        }
        
        logger.info(f"发送聊天请求到{model_name}，prompt: {prompt[:50]}...")
        response = requests.post(url, json=data)
        
        if response.status_code == 200:
            result = response.json()
            
            # 正确解析API返回格式
            if result.get("code") == 0:
                # 从data.response中获取助手回复
                assistant_response = result.get("data", {}).get("response", "")
                logger.info(f"{model_name}聊天请求成功，回复: {assistant_response[:50]}...")
                return assistant_response
            else:
                logger.error(f"{model_name} API返回错误: {result.get('message', '未知错误')}")
                return None
        else:
            logger.error(f"{model_name}聊天接口调用失败: 状态码={response.status_code}, 响应内容={response.text}")
            return None
    except Exception as e:
        logger.error(f"{model_name}聊天接口调用失败: {str(e)}")
        import traceback
        logger.error(f"异常堆栈: {traceback.format_exc()}")
        return None

def text_to_speech(text, reference_audio_file):
    """调用本地服务接口将文本转换为语音"""
    try:
        logger.info(f"发送文字转语音请求: 文本长度={len(text)}, 参考音频={reference_audio_file}")
        
        data = {
            "wav_path": reference_audio_file,
            "input_text": text,
            "output_dir": "/root/smart-yolo/MegaTTS3/output",
        }
        
        response = requests.post(TEXT_TO_SPEECH_URL, json=data)
        
        logger.info(f"收到响应: 状态码={response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            output_file = result.get('output_file')
            input_text = result.get('input_text')
            process_time = result.get('process_time')
            
            logger.info(f"文字转语音成功: 输出文件={output_file}, 处理时间={process_time}")
            
            if output_file and os.path.exists(output_file):
                with open(output_file, 'rb') as wav_file:
                    # 读取wav文件，并转换为pcm
                    audio = AudioSegment.from_wav(wav_file)
                    audio_data = audio.raw_data
                    logger.info(f"读取音频文件成功: 大小={len(audio_data)}字节")
                    return audio_data
            else:
                logger.error(f"输出文件不存在或路径无效: {output_file}")
                return None
        else:
            logger.error(f"文字转语音失败: 状态码={response.status_code}, 响应内容={response.text}")
            return None
    except Exception as e:
        logger.error(f"文字转语音接口调用失败: {str(e)}")
        import traceback
        logger.error(f"异常堆栈: {traceback.format_exc()}")
        return None

async def select_voice_category(ai_response):
    """调用chat接口，选择最适合回复的语音分类"""
    try:
        # 获取所有可用的分类名称（文件名）
        available_categories = list(AUDIO_CATEGORIES.keys())
        categories_info = ", ".join(available_categories)
        
        prompt = (
            f"请为以下回复选择最适合的语音分类。以下是可用的语音分类名称：\n"
            f"{categories_info}\n"
            f"回复内容：\n{ai_response}\n"
            f"请根据回复内容的情感、语气和上下文，选择一个最适合的语音分类名称。"
        )
        
        # 构造请求数据
        data = {
            "prompt": prompt,
            "history": conversation_history,
            "max_length": 2048,
            "temperature": 0.7,
            "top_p": 0.9
        }
        
        # 调用chat接口
        response = requests.post(CHAT_URL, json=data)
        if response.status_code == 200:
            result = response.json()
            selected_category = result.get("response", "").strip()
            
            # 检查选择的分类是否有效
            if selected_category in AUDIO_CATEGORIES:
                return selected_category
            else:
                logger.warning(f"选择的语音分类无效: {selected_category}")
                return available_categories[0] if available_categories else None
        else:
            logger.error(f"chat接口调用失败: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"选择语音分类时出错: {e}")
        return None

def use_f5tts(text, reference_audio_file):
    """调用f5tts接口将文本转换为语音"""
    from gradio_client import Client, handle_file

    # 读取reference_audio_file同名的txt文件，如果文件不存在，则使用空字符串
    reference_text_file = reference_audio_file.replace(".wav", ".txt")
    if os.path.exists(reference_text_file):
        with open(reference_text_file, "r", encoding="utf-8") as f:
            reference_text = f.read()
    else:
        reference_text = ""

    client = Client("http://127.0.0.1:7860/")
    output_audio_path, _, _, _ = client.predict(
            ref_audio_input=handle_file(reference_audio_file),
            ref_text_input=reference_text,
            gen_text_input=text,
            remove_silence=False,
            randomize_seed=True,
            seed_input=random.randint(0, 1000000),
            cross_fade_duration_slider=0.15,
            nfe_slider=32,
            speed_slider=0.8,
            api_name="/basic_tts"
    )
    if output_audio_path and os.path.exists(output_audio_path):
        with open(output_audio_path, 'rb') as wav_file:
            # 读取wav文件，并转换为pcm
            audio = AudioSegment.from_wav(wav_file)
            audio_data = audio.raw_data
            logger.info(f"读取音频文件成功: 大小={len(audio_data)}字节")
            return audio_data
    else:
        logger.error(f"输出文件不存在或路径无效: {output_audio_path}")
        return None


def process_audio(raw_audio_data, session_id):
    """处理音频数据的完整流程，支持选择不同的处理模式"""
    global reference_audio_file
    
    try:
        # 生成唯一文件名
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        wav_file_path = os.path.join(AUDIO_DIR, f"audio_input_{session_id}_{timestamp}.wav")
        
        # 将原始数据保存为WAV文件
        with wave.open(wav_file_path, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(raw_audio_data)
        logger.info(f"已保存WAV文件: {wav_file_path}")
        
        # MiniCPM模式：直接将音频发送给MiniCPM处理
        if USE_MINICPM:
            logger.info("使用MiniCPM模式处理音频...")
            # reference_audio_file = AUDIO_CATEGORIES["御姐配音暧昧"]
            output_audio_path = os.path.join(AUDIO_DIR, f"audio_output_{session_id}_{timestamp}.wav")
            text_response, audio_response, error = call_minicpm(wav_file_path, reference_audio_file, output_audio_path, session_id)
            
            if error:
                logger.error(f"MiniCPM处理失败: {error}")
                return None, f"MiniCPM处理失败: {error}"
                
            # 如果MiniCPM已经生成了音频，直接返回
            if audio_response:
                logger.info("使用MiniCPM生成的音频回复")
                return audio_response, text_response
            else:
                logger.info("正在生成语音回复...")
                if USE_F5TTS:
                    audio_response = use_f5tts(text_response, reference_audio_file)
                else:
                    audio_response = text_to_speech(text_response, reference_audio_file)
                
                # 如果成功生成语音
                if audio_response:
                    logger.info(f"已生成语音回复: {len(audio_response)} 字节")
                    return audio_response, text_response
                
                logger.warning("语音合成失败")
                return None, text_response
        
        # 常规模式：语音转文字 -> 聊天 -> 文字转语音
        else:
            # 转录音频
            logger.info("开始语音识别...")
            transcript = speech_to_text(wav_file_path)
            if not transcript:
                logger.warning("语音识别失败，未能获取文本")
                return None, "抱歉，无法识别您的语音。"
            
            logger.info(f"语音识别结果: {transcript}")
            
            # 获取聊天回复
            logger.info("正在获取AI回复...")
            ai_response = get_chat_response(transcript)
            if not ai_response:
                logger.warning("获取AI回复失败")
                return None, "抱歉，无法获取AI回复。"
            
            logger.info(f"AI回复: {ai_response}")
            
            # 如果需要跳过TTS步骤
            if SKIP_TTS:
                logger.info("跳过TTS步骤，仅返回文本回复")
                return None, ai_response
            
            # 生成语音回复
            logger.info("正在生成语音回复...")
            if USE_F5TTS:
                audio_response = use_f5tts(ai_response, reference_audio_file)
            else:
                audio_response = text_to_speech(ai_response, reference_audio_file)

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

def send_audio_chunk(ws, session_id_bytes, audio_chunk, retry_count=0):
    """发送音频数据块，带重试机制"""
    try:
        if not ws or not ws.sock or not ws.sock.connected:
            logger.error("WebSocket连接已断开，无法发送数据")
            return False
            
        # 添加会话ID前缀
        data_with_session = session_id_bytes + audio_chunk
        ws.send(data_with_session, websocket.ABNF.OPCODE_BINARY)
        return True
    except websocket.WebSocketConnectionClosedException:
        if retry_count < WS_MAX_RETRIES:
            logger.warning(f"发送数据失败，正在重试 ({retry_count + 1}/{WS_MAX_RETRIES})")
            time.sleep(WS_RECONNECT_DELAY)
            return send_audio_chunk(ws, session_id_bytes, audio_chunk, retry_count + 1)
        else:
            logger.error("达到最大重试次数，发送失败")
            return False
    except Exception as e:
        logger.error(f"发送数据时出错: {str(e)}")
        return False

def on_message(ws, message):
    """处理接收到的消息"""
    try:
        # 判断消息类型 - 文本还是二进制
        print(f"收到消息: {message}")
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
                logger.error(f"非JSON消息: {message}")
        
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
                    # ws.send(json.dumps({
                    #     "type": "text",
                    #     "session_id": session_id,
                    #     "content": "正在处理音频..."
                    # }))
                    
                    # 处理音频数据
                    audio_response, text_response = process_audio(raw_audio, session_id)
                    
                    # # 发送文本回复
                    # ws.send(json.dumps({
                    #     "type": "text",
                    #     "session_id": session_id,
                    #     "content": text_response
                    # }))
                    
                    # 发送音频回复 - 分块发送
                    if audio_response:
                        chunk_size = WS_CHUNK_SIZE
                        total_chunks = (len(audio_response) + chunk_size - 1) // chunk_size
                        
                        for i in range(0, len(audio_response), chunk_size):
                            # 截取一块音频数据
                            audio_chunk = audio_response[i:i+chunk_size]
                            # 发送数据块
                            if not send_audio_chunk(ws, session_id_bytes, audio_chunk):
                                logger.error(f"发送音频数据块失败: {i//chunk_size + 1}/{total_chunks}")
                                break
                            logger.info(f"发送音频回复块: 会话ID = {session_id}, 块大小 = {len(audio_chunk)} 字节, 进度: {i//chunk_size + 1}/{total_chunks}")
                            # 短暂暂停，避免发送过快
                            time.sleep(0.05)
                    
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

def send_heartbeat(ws):
    """发送心跳包保持连接活跃"""
    try:
        if ws and ws.sock and ws.sock.connected:
            ws.send(json.dumps({"type": "heartbeat"}))
            logger.debug("已发送心跳包")
    except Exception as e:
        logger.error(f"发送心跳包失败: {str(e)}")

def start_heartbeat(ws):
    """启动心跳线程"""
    def heartbeat_thread():
        while True:
            try:
                if ws and ws.sock and ws.sock.connected:
                    send_heartbeat(ws)
                time.sleep(WS_HEARTBEAT_INTERVAL)
            except Exception as e:
                logger.error(f"心跳线程出错: {str(e)}")
                break
    
    import threading
    heartbeat = threading.Thread(target=heartbeat_thread, daemon=True)
    heartbeat.start()
    return heartbeat

# 重连参数
max_reconnect_attempts = 10
reconnect_delay_seconds = 5
reconnect_attempt = 0

def on_error(ws, error):
    """处理错误"""
    logger.error(f"WebSocket错误: {error}")

def on_close(ws, close_status_code, close_msg):
    """处理连接关闭"""
    global reconnect_attempt, reconnect_delay_seconds, max_reconnect_attempts
    logger.warning(f"WebSocket连接已关闭: 状态码={close_status_code}, 消息={close_msg}")
    
    # 尝试重新连接
    if reconnect_attempt < max_reconnect_attempts:
        reconnect_attempt += 1
        logger.info(f"将在 {reconnect_delay_seconds} 秒后尝试重新连接...")
        time.sleep(reconnect_delay_seconds)
        # 指数退避算法增加重连延迟
        reconnect_delay_seconds = min(60, reconnect_delay_seconds * 1.5)
        start_websocket()
    else:
        logger.error(f"达到最大重连次数 ({max_reconnect_attempts})，停止重连")

def on_open(ws):
    """处理连接成功"""
    global reconnect_attempt
    reconnect_attempt = 0
    
    # 发送AI后端标识
    ws.send(json.dumps({
        "client_type": "ai_backend"
    }))
    
    logger.info("WebSocket连接已建立")

def on_ping(wsapp, message):
    print("Got a ping! A pong reply has already been automatically sent.")

def on_pong(wsapp, message):
    print("Got a pong! No need to respond")

def start_websocket():
    """启动WebSocket连接"""
    ws = websocket.WebSocketApp(
        WS_URL,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
        on_ping=on_ping,
        on_pong=on_pong
    )
    ws.on_open = on_open
    
    # 启动心跳线程
    heartbeat_thread = start_heartbeat(ws)
    
    # 设置WebSocket选项
    ws.run_forever(
        ping_interval=None,
        ping_timeout=None,
    )
    
    # 等待心跳线程结束
    if heartbeat_thread:
        heartbeat_thread.join()
            
def ai_backend_client():
    """AI后端客户端"""
    logger.info("启动WebSocket客户端")
    start_websocket()

def initialize_audio_categories():
    """初始化音频分类"""
    global AUDIO_CATEGORIES
    voice_cat_file = Path("voice_cat.json")
    
    if voice_cat_file.exists():
        # 如果voice_cat.json文件存在，直接加载分类信息
        with open(voice_cat_file, "r", encoding="utf-8") as f:
            AUDIO_CATEGORIES = json.load(f)
        logger.info("已从voice_cat.json加载音频分类信息")
    else:
        # 如果voice_cat.json文件不存在，读取音频文件并生成分类信息
        assets_dir = Path("/root/smart-yolo/MegaTTS3/assets")
        male_dir = assets_dir / "男"
        female_dir = assets_dir / "女"
        
        AUDIO_CATEGORIES = {}
        
        # 处理"男"目录下的音频文件
        if male_dir.exists() and male_dir.is_dir():
            for file_path in male_dir.glob("*.wav"):
                file_name = file_path.stem  # 获取文件名（不含扩展名）
                AUDIO_CATEGORIES[file_name] = str(file_path)
        
        # 处理"女"目录下的音频文件
        if female_dir.exists() and female_dir.is_dir():
            for file_path in female_dir.glob("*.wav"):
                file_name = file_path.stem  # 获取文件名（不含扩展名）
                AUDIO_CATEGORIES[file_name] = str(file_path)
        
        # 将分类信息保存到voice_cat.json文件
        with open(voice_cat_file, "w", encoding="utf-8") as f:
            json.dump(AUDIO_CATEGORIES, f, ensure_ascii=False, indent=4)
        logger.info("已生成并保存音频分类信息到voice_cat.json")

def main():
    # 参数解析
    parser = argparse.ArgumentParser(description="AI音频处理客户端")
    parser.add_argument("--use-minicpm", action="store_true", 
                      help="使用MiniCPM大模型进行语音处理")
    parser.add_argument("--use-qwen", action="store_true", 
                      help="使用Qwen聊天接口，而不是默认的Deepseek")
    parser.add_argument("--skip-tts", action="store_true", 
                      help="跳过文本转语音步骤")
    parser.add_argument("--use-f5tts", action="store_true", 
                      help="使用f5tts接口进行语音处理")
    parser.add_argument("--use-uncensored", action="store_true", 
                      help="使用不审查聊天接口")
    parser.add_argument("--voice-category", type=str, default="御姐配音暧昧",
                      help="指定音色名称，默认为'御姐配音暧昧'")
    
    args = parser.parse_args()
    
    # 设置全局配置
    global USE_MINICPM, USE_QWEN, SKIP_TTS, USE_F5TTS, AUDIO_DIR, PROCESSED_DIR, USE_UNCENSORED
    USE_MINICPM = args.use_minicpm
    USE_QWEN = args.use_qwen
    SKIP_TTS = args.skip_tts
    USE_F5TTS = args.use_f5tts
    USE_UNCENSORED = args.use_uncensored

    # 创建必要的目录
    setup_directories()
    
    # 初始化音频分类
    initialize_audio_categories()
    
    # 检查服务状态
    check_service_status()
    
    # 设置音色名称
    global reference_audio_file
    reference_audio_file = AUDIO_CATEGORIES.get(args.voice_category, AUDIO_CATEGORIES["御姐配音暧昧"])

    # 打印启动信息
    logger.info("=" * 50)
    logger.info("AI音频处理客户端启动")
    logger.info(f"WebSocket URL: {WS_URL}")
    logger.info(f"音频文件目录: {AUDIO_DIR}")
    logger.info(f"处理文件目录: {PROCESSED_DIR}")
    logger.info(f"使用MiniCPM: {USE_MINICPM}")
    logger.info(f"使用Qwen: {USE_QWEN}")
    logger.info(f"使用f5tts: {USE_F5TTS}")
    logger.info(f"使用不审查聊天接口: {USE_UNCENSORED}")
    logger.info(f"跳过TTS: {SKIP_TTS}")
    logger.info(f"音色名称: {args.voice_category}")
    logger.info("=" * 50)
    
    # 启动异步循环
    asyncio.run(ai_backend_client())

# 添加调用MiniCPM的函数
def call_minicpm(audio_path, reference_audio_file, output_audio_path, session_id):
    """调用MiniCPM处理音频文件，直接返回文本回复和可选的音频回复"""
    try:
        logger.info(f"开始调用MiniCPM处理音频: {audio_path}")
        
        # 检查文件是否存在
        if not os.path.exists(audio_path):
            logger.error(f"音频文件不存在: {audio_path}")
            return None, None, "音频文件不存在"
        # 获取全路径
        audio_path = os.path.abspath(audio_path)
        output_audio_path = os.path.abspath(output_audio_path)

        # 读取音频文件
        headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
        data = {
            "audio_input": audio_path,
            "ref_audio": reference_audio_file,
            "output_audio_path": output_audio_path,
            "session_id": session_id
        }
        
        logger.info(f"发送请求到MiniCPM: {MINICPM_URL}")
        
        response = requests.post(MINICPM_URL, headers=headers, json=data)

        logger.info(f"收到MiniCPM响应: 状态码={response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            if result.get("code") == 0:
                text_response = result.get("data", {}).get("text", "")
                output_audio_file_path = result.get("data", {}).get("output_audio_path", "")
                
                logger.info(f"MiniCPM处理成功，文本回复: {text_response[:50]}...")
                logger.info(f"MiniCPM生成的音频文件: {output_audio_file_path}")
                
                # 如果有音频文件，读取它
                audio_response = None
                if output_audio_file_path and os.path.exists(output_audio_file_path):
                    audio = AudioSegment.from_file(output_audio_file_path, format="wav")
                    audio_response = audio.raw_data
                
                return text_response, audio_response, None
            else:
                error_msg = result.get("message", "MiniCPM处理失败")
                logger.error(f"MiniCPM返回错误: {error_msg}")
                return None, None, error_msg
        else:
            logger.error(f"MiniCPM请求失败: 状态码={response.status_code}, 响应内容={response.text}")
            return None, None, f"MiniCPM请求失败: {response.status_code}"
    except Exception as e:
        logger.error(f"调用MiniCPM时出错: {str(e)}")
        import traceback
        logger.error(f"异常堆栈: {traceback.format_exc()}")
        return None, None, f"调用MiniCPM时出错: {str(e)}"

if __name__ == "__main__":
    main()