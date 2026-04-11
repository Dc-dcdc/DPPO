from setuptools import setup, find_packages

setup(
    name='dppo',                    # 包名
    version='0.1.0',
    packages=find_packages(),       # 自动扫描当前目录下所有带 __init__.py 的文件夹（如 env, agent 等）
    include_package_data=True,      # 允许打包非 Python 文件（如你的 XML 模型文件）
    description='DPPO Local Package',
)