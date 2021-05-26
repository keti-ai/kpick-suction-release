from ketisdk.utils.proc_utils import ProcUtils

def run():
    commands = ['demo_gui','demo_vision', 'demo_robot', 'demo_calib', 'demo_workcell']

    cmd_dict = ProcUtils().get_command_keys(commands)
    key = input('select action: ')
    input_command = cmd_dict[key]

    if input_command == 'demo_gui':
        from ketisdk.gui.gui import GUI
        GUI()
        return
    if input_command == 'demo_vision':
        from ketisdk.vision.samples.select_sample import run
    if input_command == 'demo_robot':
        from ketisdk.robot.samples.select_sample import run
    if input_command == 'demo_calib':
        from ketisdk.calib.samples.select_sample import run
    if input_command == 'demo_workcell':
        from ketisdk.workcell.samples.select_sample import run
    run()

if __name__=='__main__':
    run()



