# -*- mode: python ; coding: utf-8 -*-

import os
from PyInstaller.utils.hooks import collect_data_files

block_cipher = None

# Panda3D模型数据
panda3d_data = collect_data_files('panda3d')

# 添加自定义模型文件夹
additional_data = [
    ('models', 'models'),
    ('textures', 'textures')
]

a = Analysis(
    ['main_app.py'],
    pathex=[os.getcwd()],
    binaries=[],
    datas=panda3d_data + additional_data,
    hiddenimports=[
        'panda3d.core',
        'panda3d.egg',
        'wx.lib.pubsub'
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    name='3DChatApp',
    debug=False,
    strip=False,
    upx=True,
    runtime_tmpdir=None,
    console=False,
    icon='app_icon.ico'
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    name='3DChatApp'
)