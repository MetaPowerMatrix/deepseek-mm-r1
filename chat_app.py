// ... existing code ...
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
            logger.error("Panda3D窗口创建失败，当前图形管道: %s", pipe.getPipeType(0).getName())
            raise RuntimeError("Panda3D窗口创建失败")

        logger.info(f"使用图形管道: {self.panda_window.getPipe().getInterfaceName()}")
// ... existing code ...