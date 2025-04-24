from direct.showbase.ShowBase import ShowBase
from direct.gui.DirectGui import DirectFrame, DirectEntry, DirectButton, DirectLabel
from direct.task.Task import Task
from panda3d.core import TextNode, loadPrcFileData, CardMaker, NodePath, Texture, Vec4
from direct.gui.OnscreenImage import OnscreenImage
import logging
import os

# 配置日志系统
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='app.log'
)
logger = logging.getLogger(__name__)

class ChatApp(ShowBase):
    def __init__(self):
        loadPrcFileData("", "audio-library-name null")
        loadPrcFileData("", "gl-version 3 2")
        loadPrcFileData("", "load-file-type p3assimp")

        ShowBase.__init__(self)

        # 初始化3D场景
        self.setup_scene()

        # 初始化聊天窗口
        self.setup_chat_window()

    def setup_scene(self):
        """初始化3D场景"""
        try:
            # 设置背景图片
            background_path = os.path.join(os.path.dirname(__file__), "textures", "background.png")
            imageObject = OnscreenImage(image=background_path, pos=(0, 10, 0), parent = self.cam, scale = 3.5)
            # Note the inclusion of the "parent" keyword.
            # The scale is likely not correct; I leave it to you to find a proper value for it!
            imageObject.setBin("background", 0)
            imageObject.setDepthWrite(False)

            # image = self.loadImageAsPlane(background_path)
            # image.reparentTo(self.cam)
            # image.setTransparency(TransparencyAttrib.MAlpha)
            # self.cam2dp.node().getDisplayRegion(0).setSort(-20)

            # 设置光照
            dlight = self.render.attachNewNode("dlight")
            dlight.setPos(10, -10, 10)
            dlight.lookAt(0, 0, 0)
            # dlight.setColor((1, 1, 1, 1))  # 设置光照颜色为白色
            # dlight.setStrength(60)  # 设置光照强度
            # dlight.setShadowCaster(True)  # 启用阴影

            # 加载并设置立方体
            model_path = os.path.join(os.path.dirname(__file__), "models", "rocket.egg")
            logger.info(f"Loading model from: {model_path}")
            self.cube = self.loader.loadModel(model_path)
            self.cube.reparentTo(self.render)
            self.cube.setPos(0.5, 15, 0)
            self.cube.setScale(1)

            # 调整相机位置
            self.camera.setPos(0, -20, 10)  # 调整相机的位置
            self.camera.lookAt(0, 0, 0)  # 调整相机的视角

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
            frameSize=(-0.5, 0.5, -0.5, 0.5),  # 调整高度为窗口的一半
            frameColor=(0, 0, 0, 0.6),  # 背景设置为黑色半透明
            pos=(-0.82, 0, -0.5)  # 调整位置到窗口左下角
        )

        # 使用支持中文的字体文件
        font_path = os.path.join(os.path.dirname(__file__), "fonts", "SimHei.ttf")  # 替换为你的字体文件路径
        font = self.loader.loadFont(font_path)

        # 创建聊天显示区域
        self.chat_display = DirectLabel(
            parent=self.chat_frame,
            text="",
            text_align=TextNode.ALeft,
            text_scale=0.05,
            text_fg=(1, 1, 1, 1),
            frameColor=(0, 0, 0, 0),
            pos=(-0.48, 0, 0.45),  # 调整位置
            scale=1.0,
            text_font=font  # 设置字体文件
        )

        # 创建消息输入区域
        self.message_input = DirectEntry(
            parent=self.chat_frame,
            initialText="",
            focus=1,
            width=15,  # 调整宽度
            pos=(-0.48, 0, -0.45),  # 调整位置
            scale=0.05,
            command=self.on_send,
            text_font=font  # 设置字体文件
        )

        # 创建发送按钮
        self.send_button = DirectButton(
            parent=self.chat_frame,
            text="发送",
            scale=0.05,
            pos=(0.4, 0, -0.45),  # 调整位置
            command=self.on_send,
            text_font=font  # 设置字体文件
        )

    def on_send(self):
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