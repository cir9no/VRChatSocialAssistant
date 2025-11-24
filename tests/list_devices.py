"""
列出所有音频设备
"""

import sys
from pathlib import Path

# 添加 src 目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from audio_capture.device_manager import DeviceManager

def main():
    dm = DeviceManager()
    dm.print_device_list()
    dm.close()

if __name__ == '__main__':
    main()
