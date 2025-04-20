import wx
import wx.lib.newevent
from panda3d.core import loadPrcFileData
from direct.showbase.ShowBase import ShowBase
from direct.task.Task import Task
import os
import sys

# 自定义事件用于跨线程通信
ChatEvent, EVT_CHAT_EVENT = wx.lib.newevent.NewEvent()

class Panda3DPanel(wx.Panel):
    """嵌入Panda3D渲染的wxPython面板"""
    def __init__(self, parent, size):
        wx.Panel.__init__(self, parent, size=size)
        self.parent = parent
        
        # 设置Panda3D配置
        loadPrcFileData("", "window-type none")  # 禁用原生窗口
        loadPrcFileData("", "load-file-type p3assimp")  # 支持多种3D格式
        
        # 初始化Panda3D
        self.panda_app = ShowBase()
        self.panda_app.disableMouse()  # 禁用默认鼠标控制
        
        # 获取Panda3D的窗口句柄并嵌入到wxPython
        self.panda_window = self.panda_app.win
        self.panda_window.setWindowHandle(self.GetHandle())
        
        # 设置3D场景
        self.setup_scene()
        
        # 绑定尺寸变化事件
        self.Bind(wx.EVT_SIZE, self.on_resize)
    
    def setup_scene(self):
        """初始化3D场景"""
        # 加载环境模型
        env = self.panda_app.loader.loadModel("models/environment")
        env.reparentTo(self.panda_app.render)
        env.setScale(0.5, 0.5, 0.5)
        env.setPos(-8, 42, 0)
        
        # 添加灯光
        self.panda_app.setBackgroundColor(0.1, 0.1, 0.3)
        dlight = self.panda_app.render.attachNewNode("dlight")
        dlight.setPos(10, -10, 10)
        dlight.lookAt(0, 0, 0)
        
        # 添加一个旋转的立方体
        self.cube = self.panda_app.loader.loadModel("models/box")
        self.cube.reparentTo(self.panda_app.render)
        self.cube.setPos(0, 10, 0)
        self.cube.setScale(2)
        self.panda_app.taskMgr.add(self.rotate_cube, "rotate_cube")
    
    def rotate_cube(self, task):
        """立方体旋转动画"""
        self.cube.setHpr(task.time * 30, task.time * 40, task.time * 50)
        return Task.cont
    
    def on_resize(self, event):
        """处理窗口大小变化"""
        if self.panda_window:
            size = self.GetSize()
            self.panda_window.moveWindow(0, 0, size.width, size.height)
        event.Skip()

class ChatPanel(wx.Panel):
    """聊天对话框面板"""
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)
        self.parent = parent
        
        # 创建控件
        self.chat_display = wx.TextCtrl(self, style=wx.TE_MULTILINE|wx.TE_READONLY|wx.HSCROLL)
        self.message_input = wx.TextCtrl(self, style=wx.TE_PROCESS_ENTER)
        self.send_button = wx.Button(self, label="发送")
        
        # 设置布局
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.chat_display, proportion=1, flag=wx.EXPAND|wx.ALL, border=5)
        
        input_sizer = wx.BoxSizer(wx.HORIZONTAL)
        input_sizer.Add(self.message_input, proportion=1, flag=wx.EXPAND|wx.RIGHT, border=5)
        input_sizer.Add(self.send_button, flag=wx.EXPAND)
        
        sizer.Add(input_sizer, flag=wx.EXPAND|wx.ALL, border=5)
        self.SetSizer(sizer)
        
        # 绑定事件
        self.send_button.Bind(wx.EVT_BUTTON, self.on_send)
        self.message_input.Bind(wx.EVT_TEXT_ENTER, self.on_send)
    
    def on_send(self, event):
        """发送消息处理"""
        message = self.message_input.GetValue()
        if message:
            self.add_message("你: " + message)
            self.message_input.Clear()
            
            # 这里可以添加网络发送逻辑
            # 示例：改变3D场景作为反馈
            wx.PostEvent(self.parent, ChatEvent(message=message))
    
    def add_message(self, text):
        """添加消息到聊天显示"""
        self.chat_display.AppendText(text + "\n")

class MainWindow(wx.Frame):
    """主窗口"""
    def __init__(self):
        wx.Frame.__init__(self, None, title="3D场景与聊天", size=(1200, 700))
        
        # 创建分割窗口
        splitter = wx.SplitterWindow(self, style=wx.SP_LIVE_UPDATE)
        
        # 创建左侧3D面板 (60%宽度)
        left_panel = Panda3DPanel(splitter, size=(720, 700))
        
        # 创建右侧聊天面板 (40%宽度)
        right_panel = ChatPanel(splitter)
        
        # 设置分割比例
        splitter.SplitVertically(left_panel, right_panel, sashPosition=720)
        splitter.SetMinimumPaneSize(200)
        
        # 设置主窗口布局
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(splitter, 1, wx.EXPAND)
        self.SetSizer(sizer)
        
        # 绑定自定义事件
        self.Bind(EVT_CHAT_EVENT, self.on_chat_event)
        
        # 窗口居中显示
        self.Centre()
        self.Show()
    
    def on_chat_event(self, event):
        """处理来自聊天面板的事件"""
        message = event.message
        # 示例：根据聊天内容改变3D场景
        if "旋转" in message:
            self.GetChildren()[0].panda_app.taskMgr.remove("rotate_cube")
        elif "开始" in message:
            self.GetChildren()[0].panda_app.taskMgr.add(
                self.GetChildren()[0].rotate_cube, "rotate_cube")

if __name__ == "__main__":
    # 初始化wxPython应用
    app = wx.App(False)
    
    # 设置Panda3D模型路径
    if hasattr(sys, '_MEIPASS'):
        # 打包后的资源路径
        os.chdir(sys._MEIPASS)
    
    # 创建主窗口
    frame = MainWindow()
    
    # 主循环
    app.MainLoop()