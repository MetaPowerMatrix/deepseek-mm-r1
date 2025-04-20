import websockets
import asyncio
import json
import time
import aiohttp
from typing import Dict, Any, List, Optional

class DouyinMessageClient:
    def __init__(self, config: Dict[str, Any]):
        """
        初始化抖音消息客户端
        
        :param config: 配置字典，包含以下键:
            - douyin: 抖音API相关配置
                - client_key: 应用Key
                - client_secret: 应用密钥
                - access_token: 访问令牌
                - refresh_token: 刷新令牌(可选)
            - cloud: 云端接口配置
                - upload_url: 消息上传接口URL
                - auth_token: 认证令牌
            - websocket: WebSocket配置
                - live_ws_url: 直播间WebSocket地址
                - reconnect_interval: 重连间隔(秒)
        """
        self.config = config
        self.douyin_config = config.get('douyin', {})
        self.cloud_config = config.get('cloud', {})
        self.ws_config = config.get('websocket', {})
        
        # 会话管理
        self.session = aiohttp.ClientSession()
        self.ws_connection = None
        self.is_running = False
        
        # 消息缓存
        self.message_buffer = []
        self.buffer_size = 50  # 达到此数量后批量上传
        self.last_upload_time = 0
        
    async def connect_websocket(self):
        """连接抖音直播间WebSocket"""
        ws_url = self.ws_config.get('live_ws_url')
        if not ws_url:
            raise ValueError("WebSocket URL未配置")
            
        print(f"正在连接WebSocket: {ws_url}")
        try:
            self.ws_connection = await websockets.connect(ws_url)
            print("WebSocket连接成功")
            await self.listen_websocket()
        except Exception as e:
            print(f"WebSocket连接失败: {e}")
            await self.reconnect_websocket()
    
    async def reconnect_websocket(self):
        """WebSocket重连逻辑"""
        if not self.is_running:
            return
            
        interval = self.ws_config.get('reconnect_interval', 10)
        print(f"{interval}秒后尝试重连...")
        await asyncio.sleep(interval)
        await self.connect_websocket()
    
    async def listen_websocket(self):
        """监听WebSocket消息"""
        try:
            async for message in self.ws_connection:
                await self.process_live_message(message)
        except websockets.exceptions.ConnectionClosed:
            print("WebSocket连接已关闭")
            await self.reconnect_websocket()
        except Exception as e:
            print(f"监听WebSocket时出错: {e}")
            await self.reconnect_websocket()
    
    async def process_live_message(self, raw_message: str):
        """处理直播间公屏消息"""
        try:
            message_data = json.loads(raw_message)
            # 这里需要根据抖音实际的WebSocket消息格式进行解析
            # 以下是假设的消息格式
            message_type = message_data.get('type')
            
            if message_type == 'chat':  # 聊天消息
                sender = message_data.get('user', {}).get('nickname', '未知用户')
                content = message_data.get('content', '')
                timestamp = message_data.get('timestamp', int(time.time()))
                
                message = {
                    'type': 'live_chat',
                    'sender': sender,
                    'content': content,
                    'timestamp': timestamp,
                    'source': 'websocket'
                }
                
                print(f"[直播间消息] {sender}: {content}")
                await self.buffer_message(message)
                
        except json.JSONDecodeError:
            print(f"无法解析的消息: {raw_message}")
        except Exception as e:
            print(f"处理直播间消息时出错: {e}")
    
    async def fetch_private_messages(self, cursor: Optional[str] = None, count: int = 20):
        """获取私信消息"""
        url = "https://open.douyin.com/api/message/list"
        headers = {
            "Authorization": f"Bearer {self.douyin_config.get('access_token')}",
            "Content-Type": "application/json"
        }
        params = {
            "cursor": cursor,
            "count": count
        }
        
        try:
            async with self.session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    messages = data.get('data', {}).get('messages', [])
                    next_cursor = data.get('data', {}).get('cursor')
                    
                    for msg in messages:
                        await self.process_private_message(msg)
                    
                    # 如果有更多消息，继续获取
                    if next_cursor:
                        await asyncio.sleep(1)  # 避免请求过于频繁
                        await self.fetch_private_messages(next_cursor, count)
                        
                elif response.status == 401:  # 令牌过期
                    print("访问令牌已过期，尝试刷新...")
                    if await self.refresh_token():
                        await self.fetch_private_messages(cursor, count)
                else:
                    print(f"获取私信消息失败，状态码: {response.status}")
                    print(await response.text())
                    
        except Exception as e:
            print(f"获取私信消息时出错: {e}")
    
    async def process_private_message(self, msg_data: Dict[str, Any]):
        """处理私信消息"""
        sender = msg_data.get('sender', {}).get('nickname', '未知用户')
        content = msg_data.get('content', {}).get('text', '')
        timestamp = msg_data.get('create_time', int(time.time()))
        
        message = {
            'type': 'private_msg',
            'sender': sender,
            'content': content,
            'timestamp': timestamp,
            'source': 'api'
        }
        
        print(f"[私信消息] {sender}: {content}")
        await self.buffer_message(message)
    
    async def refresh_token(self) -> bool:
        """刷新访问令牌"""
        refresh_token = self.douyin_config.get('refresh_token')
        if not refresh_token:
            print("无刷新令牌，无法刷新访问令牌")
            return False
            
        url = "https://open.douyin.com/oauth/refresh_token"
        params = {
            "client_key": self.douyin_config.get('client_key'),
            "refresh_token": refresh_token,
            "grant_type": "refresh_token"
        }
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    new_token = data.get('data', {}).get('access_token')
                    new_refresh_token = data.get('data', {}).get('refresh_token')
                    
                    if new_token:
                        self.douyin_config['access_token'] = new_token
                        if new_refresh_token:
                            self.douyin_config['refresh_token'] = new_refresh_token
                        print("访问令牌刷新成功")
                        return True
                print(f"刷新令牌失败: {await response.text()}")
        except Exception as e:
            print(f"刷新令牌时出错: {e}")
        return False
    
    async def buffer_message(self, message: Dict[str, Any]):
        """缓冲消息，达到一定数量或时间后上传"""
        self.message_buffer.append(message)
        
        # 检查是否达到上传条件
        current_time = time.time()
        buffer_full = len(self.message_buffer) >= self.buffer_size
        time_elapsed = current_time - self.last_upload_time > 30  # 30秒上传一次
        
        if buffer_full or time_elapsed:
            await self.upload_messages()
    
    async def upload_messages(self):
        """上传消息到云端"""
        if not self.message_buffer:
            return
            
        url = self.cloud_config.get('upload_url')
        if not url:
            print("云端上传URL未配置，消息将不会被上传")
            self.message_buffer.clear()
            return
            
        headers = {
            "Authorization": f"Bearer {self.cloud_config.get('auth_token')}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "messages": self.message_buffer,
            "timestamp": int(time.time())
        }
        
        try:
            async with self.session.post(url, headers=headers, json=payload) as response:
                if response.status == 200:
                    print(f"成功上传 {len(self.message_buffer)} 条消息到云端")
                    self.message_buffer.clear()
                    self.last_upload_time = time.time()
                else:
                    print(f"上传消息失败，状态码: {response.status}")
                    print(await response.text())
                    # 上传失败保留消息，下次重试
        except Exception as e:
            print(f"上传消息时出错: {e}")
    
    async def start(self):
        """启动客户端"""
        self.is_running = True
        
        # 启动WebSocket连接
        asyncio.create_task(self.connect_websocket())
        
        # 定时获取私信消息
        while self.is_running:
            try:
                await self.fetch_private_messages()
                await asyncio.sleep(60)  # 每分钟检查一次私信
            except Exception as e:
                print(f"获取私信消息时出错: {e}")
                await asyncio.sleep(30)
    
    async def stop(self):
        """停止客户端"""
        self.is_running = False
        
        # 关闭WebSocket连接
        if self.ws_connection:
            await self.ws_connection.close()
        
        # 上传剩余消息
        if self.message_buffer:
            await self.upload_messages()
        
        # 关闭会话
        await self.session.close()
        print("客户端已停止")

async def main():
    # 配置示例 - 请替换为实际值
    config = {
        "douyin": {
            "client_key": "your_client_key",
            "client_secret": "your_client_secret",
            "access_token": "your_access_token",
            "refresh_token": "your_refresh_token"
        },
        "cloud": {
            "upload_url": "https://your.cloud.api/upload",
            "auth_token": "your_cloud_auth_token"
        },
        "websocket": {
            "live_ws_url": "wss://live.douyin.com/websocket/your_live_id",
            "reconnect_interval": 10
        }
    }
    
    client = DouyinMessageClient(config)
    try:
        await client.start()
    except KeyboardInterrupt:
        await client.stop()

if __name__ == "__main__":
    asyncio.run(main())