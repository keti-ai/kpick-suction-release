from ketisdk.gui.gui import GUI
from ketisdk.sensor.realsense_sensor import get_realsense_modules
from ketisdk.sensor.openni_sensor import get_openni_sensor_modules

def run():
    GUI(title='Realsense', modules=get_realsense_modules()+get_openni_sensor_modules())

if __name__=='__main__':
    run()
