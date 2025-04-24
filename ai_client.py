import asyncio
import json
import logging
import os
import datetime
import wave
import uuid
import argparse
import websockets
import requests
from dotenv import load_dotenv
from pathlib import Path

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
TEXT_TO_SPEECH_URL = f"{TTS_API_URL}"

# 状态接口URL
SPEECH_TO_TEXT_STATUS_URL = f"{API_URL}/speech-to-text/status"
CHAT_STATUS_URL = f"{API_URL}/chat/status"
MEGATTS_STATUS_URL = f"{API_URL}/megatts/status"

# 会话历史记录
conversation_history = []

# 全局变量
AUDIO_CATEGORIES = {}

def setup_directories():
    """确保必要的目录存在"""
    os.makedirs(AUDIO_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    logger.info(f"已创建目录: {AUDIO_DIR}, {PROCESSED_DIR}")

def check_service_status():
    """检查本地服务接口的状态"""
    try:
        # 检查语音转文字服务状态
        response = requests.get(SPEECH_TO_TEXT_STATUS_URL)
        if response.status_code == 200:
            logger.info(f"语音转文字服务状态: {response.json()}")
        else:
            logger.error(f"语音转文字服务状态检查失败: {response.status_code}")

        # # 检查聊天服务状态
        # response = requests.get(CHAT_STATUS_URL)
        # if response.status_code == 200:
        #     logger.info(f"聊天服务状态: {response.json()}")
        # else:
        #     logger.error(f"聊天服务状态检查失败: {response.status_code}")

    except Exception as e:
        logger.error(f"服务状态检查失败: {e}")

async def speech_to_text(audio_path):
    """调用本地服务接口将语音转换为文本"""
    try:
        with open(audio_path, 'rb') as audio_file:
            files = {'file': audio_file}
            response = requests.post(SPEECH_TO_TEXT_URL, files=files)
            if response.status_code == 200:
                return response.json().get("text", "")
            else:
                logger.error(f"语音转文字失败: {response.status_code}")
                return None
    except Exception as e:
        logger.error(f"语音转文字接口调用失败: {e}")
        return None

async def get_chat_response(prompt):
    """调用本地服务接口获取聊天回复"""
    try:
        data = {
            "prompt": prompt,
            "history": conversation_history,
            "max_length": 2048,
            "temperature": 0.7,
            "top_p": 0.9
        }
        response = requests.post(CHAT_URL, json=data)
        if response.status_code == 200:
            result = response.json()
            assistant_response = result.get("response", "")
            conversation_history.append({"role": "user", "content": prompt})
            conversation_history.append({"role": "assistant", "content": assistant_response})
            if len(conversation_history) > 10:
                conversation_history = conversation_history[-10:]
            return assistant_response
        else:
            logger.error(f"聊天接口调用失败: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"聊天接口调用失败: {e}")
        return None

async def text_to_speech(text, reference_audio_file):
    """调用本地服务接口将文本转换为语音"""
    try:
        data = {
            "wav_path": reference_audio_file,
            "input_text": text,
            "output_dir": "/root/smart-yolo/MegaTTS3/output",
        }
        response = requests.post(TEXT_TO_SPEECH_URL, json=data)
        if response.status_code == 200:
            # 返回json数据{'output_file': output_file},读取这个wav文件转为PCM音频数据返回
            output_file = response.json().get('output_file')
            with open(output_file, 'rb') as wav_file:
                return wav_file.read()
        else:
            logger.error(f"文字转语音失败: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"文字转语音接口调用失败: {e}")
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

async def process_audio(raw_audio_data, session_id):
    """处理音频数据的完整流程"""
    temp_files = []  # 记录临时文件以便清理
    
    try:
        # 生成唯一文件名
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        wav_file_path = os.path.join(AUDIO_DIR, f"audio_{session_id}_{timestamp}.wav")
        temp_files.append(wav_file_path)
        
        # 将原始数据保存为WAV文件
        with wave.open(wav_file_path, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(raw_audio_data)
        logger.info(f"已保存WAV文件: {wav_file_path}")
        
        # 转录音频
        logger.info("开始语音识别...")
        transcript = await speech_to_text(wav_file_path)
        if not transcript:
            logger.warning("语音识别失败，未能获取文本")
            return None, "抱歉，无法识别您的语音。"
        
        logger.info(f"语音识别结果: {transcript}")
        
        # 获取聊天回复
        logger.info("正在获取AI回复...")
        ai_response = await get_chat_response(transcript)
        if not ai_response:
            logger.warning("获取AI回复失败")
            return None, "抱歉，无法获取AI回复。"
        
        logger.info(f"AI回复: {ai_response}")

        # 选择最适合的语音分类
        selected_category = await select_voice_category(ai_response)
        if selected_category:
            logger.info(f"选择的语音分类: {selected_category}")
        else:
            logger.warning("未能选择有效的语音分类，使用默认分类")
            selected_category = list(AUDIO_CATEGORIES.keys())[0]  # 默认使用第一个分类

        # 根据分类获取参考音频文件
        reference_audio_file = AUDIO_CATEGORIES[selected_category]
        
        # 生成语音回复
        logger.info("正在生成语音回复...")
        audio_response = await text_to_speech(ai_response, reference_audio_file)

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
    parser.add_argument("--audio-dir", type=str, default="audio_files",
                      help="音频文件存储目录")
    parser.add_argument("--processed-dir", type=str, default="processed_files",
                      help="处理后文件存储目录")
    
    # args = parser.parse_args()
    
    # 创建必要的目录
    setup_directories()
    
    # 初始化音频分类
    initialize_audio_categories()
    
    # 检查服务状态
    check_service_status()
    
    # 打印启动信息
    logger.info("=" * 50)
    logger.info("AI音频处理客户端启动")
    logger.info(f"WebSocket URL: {WS_URL}")
    logger.info(f"音频文件目录: {AUDIO_DIR}")
    logger.info(f"处理文件目录: {PROCESSED_DIR}")
    logger.info("=" * 50)
    
    # 启动异步循环
    asyncio.run(ai_backend_client(WS_URL))

if __name__ == "__main__":
    main()