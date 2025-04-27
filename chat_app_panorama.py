from direct.showbase.ShowBase import ShowBase
from panda3d.core import *
from panda3d.core import loadPrcFileData
from direct.gui.DirectGui import *
import os
import math
from pathlib import Path
import gltf

# 配置抗锯齿
loadPrcFileData("", "multisamples 4")
loadPrcFileData("", "framebuffer-multisample 1")
loadPrcFileData("", "audio-library-name null")
loadPrcFileData("", "gl-version 3 2")
loadPrcFileData("", "load-file-type p3assimp")

current_dir = Path(__file__).parent

class DAEViewer():
    def __init__(self, scene):
        # 加载DAE模型
        try:
            model_robot = os.path.join(os.path.dirname(__file__), "models", "alex")
            print(f"Loading model from: {model_robot}")
            self.model = scene.loader.load_model(Filename(model_robot, "alex.gltf"))
            self.model.reparentTo(scene.render)
            self.model.setPos(0, 10, 0)
            self.model.setScale(0.5)
            
            # 设置光照
            self.setup_lights(scene)
            
            # 播放动画
            # self.setup_animation()
            
        except Exception as e:
            print("加载gltf模型失败:", str(e))
    
    def setup_lights(self, scene):
        """设置基本光照"""
        ambient = AmbientLight("ambient")
        ambient.setColor((0.2, 0.2, 0.2, 1))
        scene.render.setLight(scene.render.attachNewNode(ambient))
        
        dlight = DirectionalLight("dlight")
        dlight.setColor((0.8, 0.8, 0.8, 1))
        dlnp = scene.render.attachNewNode(dlight)
        dlnp.setHpr(45, -60, 0)
        scene.render.setLight(dlnp)
    
    # def setup_animation(self):
    #     """设置并播放动画"""
    #     anims = self.model.getAnimControls()
    #     if anims:
    #         print(f"找到 {len(anims)} 个动画")
    #         for anim in anims:
    #             print(" -", anim.name)
    #             anim.loop()  # 循环播放第一个动画
    #     else:
    #         print("没有找到动画")

class AdvancedPanoramaViewer(ShowBase):
    def __init__(self, panorama_path):
        ShowBase.__init__(self)
        
        # 使用支持中文的字体文件
        font_path = os.path.join(os.path.dirname(__file__), "fonts", "SimHei.ttf")  # 替换为你的字体文件路径
        self.font = self.loader.loadFont(font_path)

        # 初始化设置
        self.setup_render()
        self.create_sphere(100, 64, 64)
        self.load_panorama(panorama_path)
        self.setup_controls()
        self.setup_ui()
        
        # 视角参数
        self.rotation_speed = 0
        self.pitch_speed = 0
        self.fov = 60
        self.zoom_speed = 0.1
        self.inertia = 0.9  # 惯性系数
        
        # 初始视角
        self.camera.setHpr(0, 0, 0)
        self.camLens.setFov(self.fov)

        self.viewer = DAEViewer(self)
    
    def create_sphere(self, radius, slices, stacks):
        """程序化创建高精度球体"""
        format = GeomVertexFormat.getV3n3t2()
        vdata = GeomVertexData("sphere", format, Geom.UHStatic)
        
        vertex = GeomVertexWriter(vdata, "vertex")
        normal = GeomVertexWriter(vdata, "normal")
        texcoord = GeomVertexWriter(vdata, "texcoord")
        
        geom = Geom(vdata)
        tris = GeomTriangles(Geom.UHStatic)
        
        # 生成顶点
        for i in range(stacks + 1):
            v = i / float(stacks)
            phi = v * math.pi
            
            for j in range(slices + 1):
                u = j / float(slices)
                theta = u * math.pi * 2
                
                x = math.sin(phi) * math.cos(theta)
                y = math.sin(phi) * math.sin(theta)
                z = math.cos(phi)
                
                vertex.addData3(x * radius, y * radius, z * radius)
                normal.addData3(x, y, z)
                texcoord.addData2(u, 1 - v)
        
        # 生成三角形
        for i in range(stacks):
            for j in range(slices):
                a = i * (slices + 1) + j
                b = a + slices + 1
                c = b + 1
                d = a + 1
                
                tris.addVertices(a, b, c)
                tris.addVertices(a, c, d)
        
        geom.addPrimitive(tris)
        node = GeomNode("sphere")
        node.addGeom(geom)
        
        self.sphere = self.render.attachNewNode(node)
        self.sphere.setTwoSided(True)
        self.sphere.setBin("background", 1)
        self.sphere.setDepthWrite(False)
    
    def setup_render(self):
        """配置渲染设置"""
        self.disableMouse()
        self.setBackgroundColor(0, 0, 0)
        self.render.setShaderAuto()
        
        # 提高纹理质量
        tex = Texture()
        tex.setMinfilter(SamplerState.FTLinearMipmapLinear)
        tex.setMagfilter(SamplerState.FTLinear)
    
    def load_panorama(self, path):
        """加载全景图片"""
        # 加载并应用纹理
        tex = self.loader.loadTexture(path)
        tex.setWrapU(SamplerState.WMRepeat)
        tex.setWrapV(SamplerState.WMClamp)
        self.sphere.setTexture(tex, 1)
        
        # 反转法线使纹理在内部可见
        self.sphere.setTwoSided(True)
        self.sphere.setBin("background", 1)
        self.sphere.setDepthWrite(False)
        self.sphere.reparentTo(self.render)
        
        # 将相机放在中心
        self.camera.setPos(0, 0, 0)
    
    def setup_controls(self):
        """设置输入控制"""
        # 鼠标控制
        self.accept("mouse1", self.start_drag)
        self.accept("mouse1-up", self.stop_drag)
        self.accept("wheel_up", self.zoom_in)
        self.accept("wheel_down", self.zoom_out)
        
        # 键盘控制
        self.accept("arrow_left", self.set_rotation_speed, [-1])
        self.accept("arrow_left-up", self.set_rotation_speed, [0])
        self.accept("arrow_right", self.set_rotation_speed, [1])
        self.accept("arrow_right-up", self.set_rotation_speed, [0])
        self.accept("arrow_up", self.set_pitch_speed, [1])
        self.accept("arrow_up-up", self.set_pitch_speed, [0])
        self.accept("arrow_down", self.set_pitch_speed, [-1])
        self.accept("arrow_down-up", self.set_pitch_speed, [0])
        self.accept("f1", self.reset_view)
        self.accept("f2", self.toggle_fullscreen)
        
        # 添加更新任务
        self.taskMgr.add(self.update_view, "update_view")
    
    def setup_ui(self):
        """设置用户界面"""
        self.title = OnscreenText(
            text="360°全景查看器 - 使用鼠标拖拽查看，滚轮缩放",
            pos=(0, 0.9),
            scale=0.06,
            fg=(1, 1, 1, 1),
            align=TextNode.ACenter,
            mayChange=False,
            font=self.font
        )
    
    def start_drag(self):
        """开始拖拽"""
        if self.mouseWatcherNode.hasMouse():
            self.lastX = self.mouseWatcherNode.getMouseX()
            self.lastY = self.mouseWatcherNode.getMouseY()
            self.taskMgr.add(self.drag_camera, "drag_camera")
    
    def stop_drag(self):
        """停止拖拽"""
        self.taskMgr.remove("drag_camera")
    
    def drag_camera(self, task):
        """处理拖拽移动"""
        if self.mouseWatcherNode.hasMouse():
            currentX = self.mouseWatcherNode.getMouseX()
            currentY = self.mouseWatcherNode.getMouseY()
            
            dx = currentX - self.lastX
            dy = currentY - self.lastY
            
            # 设置旋转速度（用于惯性）
            self.rotation_speed = -dx * 50
            self.pitch_speed = dy * 50
            
            self.lastX = currentX
            self.lastY = currentY
        
        return task.cont
    
    def set_rotation_speed(self, speed):
        """设置水平旋转速度"""
        self.rotation_speed = speed * 30
    
    def set_pitch_speed(self, speed):
        """设置垂直旋转速度"""
        self.pitch_speed = speed * 30
    
    def zoom_in(self):
        """放大视图"""
        self.fov = max(10, self.fov - 5)
        self.camLens.setFov(self.fov)
    
    def zoom_out(self):
        """缩小视图"""
        self.fov = min(120, self.fov + 5)
        self.camLens.setFov(self.fov)
    
    def reset_view(self):
        """重置视角"""
        self.camera.setHpr(0, 0, 0)
        self.fov = 60
        self.camLens.setFov(self.fov)
        self.rotation_speed = 0
        self.pitch_speed = 0
    
    def toggle_fullscreen(self):
        """切换全屏"""
        props = self.win.getProperties()
        self.win.requestProperties(
            WindowProperties(fullscreen=not props.getFullscreen()))
    
    def update_view(self, task):
        """更新视角"""
        # 应用惯性
        self.rotation_speed *= self.inertia
        self.pitch_speed *= self.inertia
        
        # 更新相机方向
        hpr = self.camera.getHpr()
        newH = hpr.x + self.rotation_speed * globalClock.getDt()
        newP = max(-90, min(90, hpr.y + self.pitch_speed * globalClock.getDt()))
        self.camera.setHpr(newH, newP, 0)
        
        # 如果速度很小，则停止
        if abs(self.rotation_speed) < 0.1:
            self.rotation_speed = 0
        if abs(self.pitch_speed) < 0.1:
            self.pitch_speed = 0
        
        return task.cont

if __name__ == "__main__":
    # 替换为你的全景图片路径
    panorama_path = "./textures/yulanlu.jpeg"
    
    viewer = AdvancedPanoramaViewer(panorama_path)
    viewer.run()