from pyautd3 import AUTD3

# AUTDの配列を定義する

# 6台のAUTDを3行2列の配置で使用する
# 05
# 14
# 23
class AutdArrangement:
    def __init__(self):
        self.autd_arrangement = [
            AUTD3(pos=[0.0, 0.0, 0.0], rot=[1, 0, 0, 0]), 
            AUTD3(pos=[0.0, -(AUTD3.DEVICE_HEIGHT), 0.0], rot=[1, 0, 0, 0]),
            AUTD3(pos=[0.0, -2 * (AUTD3.DEVICE_HEIGHT), 0.0], rot=[1, 0, 0, 0]),
            AUTD3(pos=[AUTD3.DEVICE_WIDTH, -2 * (AUTD3.DEVICE_HEIGHT), 0.0], rot=[1, 0, 0, 0]),
            AUTD3(pos=[AUTD3.DEVICE_WIDTH, -(AUTD3.DEVICE_HEIGHT), 0.0], rot=[1, 0, 0, 0]),
            AUTD3(pos=[AUTD3.DEVICE_WIDTH, 0.0, 0.0], rot=[1, 0, 0, 0]),
        ]
        
