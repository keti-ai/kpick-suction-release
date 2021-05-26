from ketisdk.utils.proc_utils import ProcUtils

def run():
    commands = ['run_realsense_gui', 'demo_client_gui']

    cmd_dict = ProcUtils().get_command_keys(commands)
    key = input('select action: ')
    input_command = cmd_dict[key]

    if input_command == 'run_sensors':
        from .run_sensors import run
    if input_command == 'demo_client_gui':
        from .demo_client_gui import demo_client_gui as run

    run()

if __name__=='__main__':
    run()



