from direct.showbase.ShowBase import ShowBase
from direct.gui.DirectGui import DirectFrame, DirectEntry, DirectButton, DirectLabel
from direct.task.Task import Task
from panda3d.core import TextNode, loadPrcFileData
import logging
from pathlib import Path

# 配置日志系统
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='app.log'
)
logger = logging.getLogger(__name__)

# 设置 OpenGL 版本为 2.1
loadPrcFileData("", "gl-version 2 1")

# 使用离屏渲染
loadPrcFileData("", "window-type offscreen")

class ChatApp(ShowBase):
    def __init__(self):
        loadPrcFileData("", "audio-library-name null")

        ShowBase.__init__(self)

        # 初始化3D场景
        self.setup_scene()

        # 初始化聊天窗口
        self.setup_chat_window()

    def setup_scene(self):
        """初始化3D场景"""
        try:
            # 设置背景颜色
            self.setBackgroundColor(0.1, 0.1, 0.3)

            # 加载并设置立方体
            self.cube = self.loader.loadModel("models/environment.glb")
            self.cube.reparentTo(self.render)
            self.cube.setPos(0, 10, 0)
            self.cube.setScale(2)

            # 启动动画任务
            self.taskMgr.add(self.rotate_cube, "rotate_cube")

            logger.info("3D场景初始化完成")
        except Exception as e:
            logger.error(f"场景加载失败: {str(e)}", exc_info=True)
            self._create_placeholder_geometry()

    def _create_placeholder_geometry(self):
        """创建替代几何体"""
        from panda3d.core import CardMaker
        cm = CardMaker("placeholder")
        cm.setFrame(-1, 1, -1, 1)
        card = self.render.attachNewNode(cm.generate())
        card.setColor(1, 0, 0, 1)  # 红色作为错误指示
        logger.warning("使用替代几何体")

    def rotate_cube(self, task):
        """立方体旋转动画（线程安全）"""
        if hasattr(self, 'cube') and self.cube:
            self.cube.setHpr(task.time * 30, task.time * 40, task.time * 50)
        return Task.cont

    def setup_chat_window(self):
        """初始化聊天窗口"""
        # 创建聊天窗口背景
        self.chat_frame = DirectFrame(
            frameSize=(-0.3, 0.3, -0.4, 0.1),
            frameColor=(0.1, 0.1, 0.1, 0.8),
            pos=(0.7, 0, -0.5)
        )

        # 创建聊天显示区域
        self.chat_display = DirectLabel(
            parent=self.chat_frame,
            text="",
            text_align=TextNode.ALeft,
            text_scale=0.05,
            text_fg=(1, 1, 1, 1),
            frameColor=(0, 0, 0, 0),
            pos=(-0.28, 0, 0.05),
            scale=1.0
        )

        # 创建消息输入区域
        self.message_input = DirectEntry(
            parent=self.chat_frame,
            initialText="",
            focus=1,
            width=20,
            pos=(-0.28, 0, -0.1),
            scale=0.05,
            command=self.on_send
        )

        # 创建发送按钮
        self.send_button = DirectButton(
            parent=self.chat_frame,
            text="发送",
            scale=0.05,
            pos=(0.2, 0, -0.1),
            command=self.on_send
        )

    def on_send(self, text):
        """处理消息发送"""
        message = self.message_input.get()
        if not message:
            return

        # 添加消息到聊天显示区域
        self.chat_display["text"] += f"你: {message}\n"
        self.message_input.enterText("")

        # 处理3D场景控制命令
        self.process_3d_command(message)

    def process_3d_command(self, message):
        """处理3D场景控制命令"""
        try:
            if "旋转" in message:
                self.taskMgr.remove("rotate_cube")
                self.chat_display["text"] += "系统: 已停止立方体旋转\n"
            elif "开始" in message:
                self.taskMgr.add(self.rotate_cube, "rotate_cube")
                self.chat_display["text"] += "系统: 立方体开始旋转\n"
        except Exception as e:
            logger.error(f"3D命令处理失败: {str(e)}", exc_info=True)

if __name__ == "__main__":
    app = ChatApp()
    app.run()