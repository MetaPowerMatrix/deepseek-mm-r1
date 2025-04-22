import wx
import wx.lib.newevent
from panda3d.core import loadPrcFileData, GraphicsPipeSelection
from direct.showbase.ShowBase import ShowBase
from direct.task.Task import Task
import os
import sys
import logging
from pathlib import Path

# 配置日志系统
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='app.log'
)
logger = logging.getLogger(__name__)

# 自定义事件用于跨线程通信
ChatEvent, EVT_CHAT_EVENT = wx.lib.newevent.NewEvent()

class Panda3DPanel(wx.Panel):
    """改进后的Panda3D渲染面板，包含健壮的错误处理和资源管理"""
    def __init__(self, parent, size):
        wx.Panel.__init__(self, parent, size=size)
        self.parent = parent
        self.panda_app = None
        self.panda_window = None
        self.models_loaded = False

        try:
            wx.CallAfter(self._initialize_panda3d)
            self._verify_resources
            wx.CallAfter(self._finalize_setup)
        except Exception as e:
            logger.error(f"初始化失败: {str(e)}", exc_info=True)
            wx.MessageBox(
                f"3D引擎初始化失败: {str(e)}", 
                "严重错误",
                wx.OK | wx.ICON_ERROR
            )
            raise

    def _initialize_panda3d(self):
        """分步初始化Panda3D引擎"""
        # 关键配置必须放在ShowBase之前
        loadPrcFileData("", "window-type none")
        loadPrcFileData("", "load-file-type p3assimp")
        loadPrcFileData("", "audio-library-name null")
        loadPrcFileData("", "gl-version 3 2")
        loadPrcFileData("", "framebuffer-multisample 1")
        loadPrcFileData("", "multisamples 4")

        # 验证图形管道
        pipe = GraphicsPipeSelection.getGlobalPtr()
        if pipe.getNumPipeTypes() == 0:
            raise RuntimeError("没有可用的图形管道，请检查显卡驱动")

        self.panda_app = ShowBase()
        self.panda_app.disableMouse()
        self.panda_window = self.panda_app.win

        if self.panda_window is None:
            raise RuntimeError("Panda3D窗口创建失败")

        logger.info(f"使用图形管道: {self.panda_window.getPipe().getInterfaceName()}")

    def _verify_resources(self):
        """验证模型资源是否存在"""
        self.model_dir = Path(__file__).parent / "models"
        required_models = ["crossroad.glb", "alex.obj"]
        
        for model in required_models:
            if not (self.model_dir / model).exists():
                raise FileNotFoundError(
                    f"模型文件缺失: {self.model_dir/model}\n"
                    f"请确保models目录包含: {', '.join(required_models)}"
                )

    def _finalize_setup(self):
        """窗口显示后完成初始化"""
        if not self.panda_window.setWindowHandle(self.GetHandle()):
            logger.warning("窗口句柄设置失败，尝试备用方法")
            self._setup_fallback_render()

        self.setup_scene()
        self.Bind(wx.EVT_SIZE, self.on_resize)
        self.models_loaded = True

    def _setup_fallback_render(self):
        """备用渲染方案"""
        logger.info("使用帧缓冲对象(FBO)离屏渲染")
        self.buffer = self.panda_app.win.makeTextureBuffer(
            "wx_buffer", 
            self.GetSize().width,
            self.GetSize().height
        )
        self.texture = self.buffer.getTexture()
        self.panda_app.cam.reparentTo(self.buffer.makeCamera())

    def setup_scene(self):
        """安全地初始化3D场景"""
        try:
            # 加载环境模型
            env_path = str(self.model_dir / "crossroad.glb")
            env = self.panda_app.loader.loadModel(env_path)
            env.reparentTo(self.panda_app.render)
            env.setScale(0.5, 0.5, 0.5)
            env.setPos(-8, 42, 0)
            
            # 设置灯光和背景
            self.panda_app.setBackgroundColor(0.1, 0.1, 0.3)
            dlight = self.panda_app.render.attachNewNode("dlight")
            dlight.setPos(10, -10, 10)
            dlight.lookAt(0, 0, 0)
            
            # 加载并设置立方体
            box_path = str(self.model_dir / "alex.obj")
            self.cube = self.panda_app.loader.loadModel(box_path)
            self.cube.reparentTo(self.panda_app.render)
            self.cube.setPos(0, 10, 0)
            self.cube.setScale(1)
            
            # 启动动画任务
            self.panda_app.taskMgr.add(self.rotate_cube, "rotate_cube")
            
            logger.info("3D场景初始化完成")
        except Exception as e:
            logger.error(f"场景加载失败: {str(e)}", exc_info=True)
            self._create_placeholder_geometry()

    def _create_placeholder_geometry(self):
        """创建替代几何体"""
        from panda3d.core import CardMaker
        cm = CardMaker("placeholder")
        cm.setFrame(-1, 1, -1, 1)
        card = self.panda_app.render.attachNewNode(cm.generate())
        card.setColor(1, 0, 0, 1)  # 红色作为错误指示
        logger.warning("使用替代几何体")

    def rotate_cube(self, task):
        """立方体旋转动画（线程安全）"""
        if hasattr(self, 'cube') and self.cube:
            self.cube.setHpr(task.time * 30, task.time * 40, task.time * 50)
        return Task.cont
    
    def on_resize(self, event):
        """安全处理窗口大小变化"""
        if self.panda_window and self.GetSize().width > 0:
            size = self.GetClientSize()
            self.panda_window.moveWindow(0, 0, size.width, size.height)
            
            # 更新相机视口
            lens = self.panda_app.cam.node().getLens()
            lens.setAspectRatio(size.width / max(1, size.height))
            
        event.Skip()

    def __del__(self):
        """确保资源清理"""
        if hasattr(self, 'panda_app') and self.panda_app:
            self.panda_app.destroy()
            logger.info("Panda3D资源已释放")

class ChatPanel(wx.Panel):
    """增强的聊天面板，支持历史记录和输入验证"""
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)
        self.parent = parent
        self.history = []
        
        self._setup_ui()
        self._bind_events()
        
    def _setup_ui(self):
        """初始化UI组件"""
        # 聊天显示区域
        self.chat_display = wx.TextCtrl(
            self, 
            style=wx.TE_MULTILINE | wx.TE_READONLY | wx.HSCROLL | wx.TE_RICH2
        )
        self.chat_display.SetBackgroundColour(wx.Colour(240, 240, 240))
        
        # 消息输入区域
        self.message_input = wx.TextCtrl(
            self,
            style=wx.TE_PROCESS_ENTER | wx.TE_MULTILINE
        )
        self.message_input.SetHint("输入消息...")
        
        # 发送按钮
        self.send_button = wx.Button(self, label="发送")
        self.send_button.SetBackgroundColour(wx.Colour(100, 160, 220))
        self.send_button.SetForegroundColour(wx.WHITE)
        
        # 布局设置
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.chat_display, proportion=1, flag=wx.EXPAND | wx.ALL, border=5)
        
        input_sizer = wx.BoxSizer(wx.HORIZONTAL)
        input_sizer.Add(self.message_input, proportion=1, flag=wx.EXPAND | wx.RIGHT, border=5)
        input_sizer.Add(self.send_button, flag=wx.EXPAND)
        
        sizer.Add(input_sizer, flag=wx.EXPAND | wx.ALL, border=5)
        self.SetSizer(sizer)
        
    def _bind_events(self):
        """绑定事件处理"""
        self.send_button.Bind(wx.EVT_BUTTON, self.on_send)
        self.message_input.Bind(wx.EVT_TEXT_ENTER, self.on_send)
        self.message_input.Bind(wx.EVT_CHAR, self._on_key_press)

    def _on_key_press(self, event):
        """处理特殊按键"""
        # Ctrl+Enter 换行，Enter 发送
        if event.ControlDown() and event.GetKeyCode() == wx.WXK_RETURN:
            self.message_input.WriteText("\n")
        else:
            event.Skip()

    def on_send(self, event):
        """安全处理消息发送"""
        message = self.message_input.GetValue().strip()
        if not message:
            return
            
        try:
            self.add_message("你: " + message, is_user=True)
            self.message_input.Clear()
            
            # 通过事件机制通知主窗口
            wx.PostEvent(self.parent, ChatEvent(message=message))
        except Exception as e:
            logger.error(f"消息处理失败: {str(e)}", exc_info=True)
            wx.MessageBox("消息发送失败", "错误", wx.OK | wx.ICON_ERROR)

    def add_message(self, text, is_user=False):
        """添加带格式的消息"""
        self.history.append(text)
        
        # 设置不同消息类型的样式
        self.chat_display.SetDefaultStyle(
            wx.TextAttr(
                wx.BLUE if is_user else wx.BLACK,
                wx.NullColour,
                wx.Font(10, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, 
                       wx.FONTWEIGHT_BOLD if is_user else wx.FONTWEIGHT_NORMAL)
            )
        )
        self.chat_display.AppendText(text + "\n")
        
        # 自动滚动到底部
        self.chat_display.ShowPosition(self.chat_display.GetLastPosition())

class MainWindow(wx.Frame):
    """主窗口，负责协调3D视图和聊天面板"""
    def __init__(self):
        wx.Frame.__init__(
            self, 
            None, 
            title="3D场景与聊天系统",
            size=(1200, 700),
            style=wx.DEFAULT_FRAME_STYLE & ~(wx.RESIZE_BORDER | wx.MAXIMIZE_BOX)
        )
        
        self._setup_ui()
        self._setup_menu()
        self._bind_events()
        
        # 窗口居中并显示
        self.Centre()
        self.Show()
        
        # 初始状态检查
        self._check_initialization()

    def _setup_ui(self):
        """初始化用户界面"""
        # 创建分割窗口
        self.splitter = wx.SplitterWindow(self, style=wx.SP_LIVE_UPDATE)
        
        # 左侧3D面板 (60%宽度)
        self.left_panel = Panda3DPanel(self.splitter, size=(720, 700))
        
        # 右侧聊天面板 (40%宽度)
        self.right_panel = ChatPanel(self.splitter)
        
        # 设置分割比例
        self.splitter.SplitVertically(self.left_panel, self.right_panel, sashPosition=720)
        self.splitter.SetMinimumPaneSize(200)
        
        # 主布局
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(self.splitter, 1, wx.EXPAND)
        self.SetSizer(sizer)

    def _setup_menu(self):
        """创建菜单栏"""
        menubar = wx.MenuBar()
        
        # 文件菜单
        file_menu = wx.Menu()
        exit_item = file_menu.Append(wx.ID_EXIT, "退出", "退出应用程序")
        menubar.Append(file_menu, "文件")
        
        # 帮助菜单
        help_menu = wx.Menu()
        about_item = help_menu.Append(wx.ID_ABOUT, "关于", "程序信息")
        menubar.Append(help_menu, "帮助")
        
        self.SetMenuBar(menubar)

    def _bind_events(self):
        """绑定事件处理"""
        self.Bind(EVT_CHAT_EVENT, self.on_chat_event)
        self.Bind(wx.EVT_MENU, self.on_exit, id=wx.ID_EXIT)
        self.Bind(wx.EVT_MENU, self.on_about, id=wx.ID_ABOUT)
        self.Bind(wx.EVT_CLOSE, self.on_close)

    def _check_initialization(self):
        """验证初始化状态"""
        if not hasattr(self.left_panel, 'models_loaded') or not self.left_panel.models_loaded:
            wx.MessageBox(
                "3D场景初始化不完整，部分功能可能受限",
                "警告",
                wx.OK | wx.ICON_WARNING
            )

    def on_chat_event(self, event):
        """线程安全地处理聊天事件"""
        wx.CallAfter(self._process_3d_command, event.message)

    def _process_3d_command(self, message):
        """处理3D场景控制命令"""
        try:
            if "旋转" in message:
                self.left_panel.panda_app.taskMgr.remove("rotate_cube")
                self.right_panel.add_message("系统: 已停止立方体旋转", is_user=False)
            elif "开始" in message:
                self.left_panel.panda_app.taskMgr.add(
                    self.left_panel.rotate_cube, 
                    "rotate_cube"
                )
                self.right_panel.add_message("系统: 立方体开始旋转", is_user=False)
        except Exception as e:
            logger.error(f"3D命令处理失败: {str(e)}", exc_info=True)

    def on_about(self, event):
        """显示关于对话框"""
        info = wx.adv.AboutDialogInfo()
        info.SetName("3D聊天系统")
        info.SetVersion("1.0")
        info.SetDescription(
            "集成3D场景与实时聊天的演示程序\n"
            "使用技术: wxPython + Panda3D"
        )
        info.SetCopyright("(C) 2024")
        wx.adv.AboutBox(info)

    def on_exit(self, event):
        """安全退出"""
        self.Close()

    def on_close(self, event):
        """关闭前清理资源"""
        if hasattr(self.left_panel, 'panda_app'):
            self.left_panel.panda_app.destroy()
        self.Destroy()

def main():
    """应用程序入口点"""
    # Windows平台高DPI支持
    if sys.platform == 'win32':
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    
    app = wx.App()
    app.SetAppName("3DChatSystem")
    
    try:
        # 检查资源路径
        if hasattr(sys, '_MEIPASS'):
            os.chdir(sys._MEIPASS)
            
        frame = MainWindow()
        app.MainLoop()
    except Exception as e:
        logger.critical(f"应用程序崩溃: {str(e)}", exc_info=True)
        wx.MessageBox(
            f"程序遇到致命错误:\n{str(e)}\n\n详见日志文件: {os.path.abspath('app.log')}",
            "致命错误",
            wx.OK | wx.ICON_ERROR
        )
        sys.exit(1)

if __name__ == "__main__":
    main()