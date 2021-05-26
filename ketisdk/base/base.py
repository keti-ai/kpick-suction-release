import os
from shutil import rmtree
from glob import glob
from ..utils.proc_utils import ProcUtils, CFG

class BasObj():
    """ basic object:

    - initialize from **args** or **cfg_path**

    :param args: configuration arguments
    :type args: :class:`.CFG`
    :param cfg_path: configuration file path
    :type cfg_path: .cfg file

    *Example*: objA = BasObj(cfg_path='a_config_file.cfg')
    """
    def __init__(self, args=None, cfg_path=None, name='unnamed'):
        self.cfg_path = cfg_path
        self.name = name
        self.args = args
        if args is None and cfg_path is None:
            print('{} Object initialzed without args'.format('+'*10))
            return
        if ProcUtils().isexists(cfg_path): args = CFG(cfg_path=cfg_path)
        self.load_params(args=args)

    def load_params(self, args):
        """ load arguments """
        self.args = args

    def reload_params(self, cfg_path=None):
        """ in-excution reload arguments from cfg_path """
        if cfg_path is None: cfg_path = self.cfg_path
        if not ProcUtils().isexists(cfg_path):
            if self.args is not None: self.load_params(args=self.args)
            else: print('%s does not exist >>> use existing configs ...')
        else:
            args = CFG(cfg_path=cfg_path)
            self.load_params(args=args)
            print('configs reloaded ....')

class BasFileManage():
    """ basic file manager """
    def move_files_dir2dir(self, in_dir, out_dir, file_format='*'):
        """ move all file with `format` from `in_dir` to `out_dir`

        :param in_dir: input directory path
        :type in_dir: str
        :param out_dir: output directory path
        :type out_dir: str
        :param file_format: format of which is going to be move
        :type file_format: str
        """
        assert os.path.exists(in_dir)
        if not os.path.exists(out_dir): os.makedirs(out_dir)

        file_paths = glob(os.path.join(in_dir, file_format), recursive=True)

        for path in file_paths:
            print('moving %s ...' %path)
            os.rename(path, path.replace(in_dir, out_dir))

    # def add_subdir_to_path(self, path):
    #     indir, filename = os.path.split(path)
    #     indir, subdir = os.path.split(indir)
    #     return os.path.join(indir, '%s_%s' %(subdir, filename))

    def move_file(self, path, root_dir, out_dir):
        """ move a file with given `path` from  `root_dir` to `out_dir`.
        Add subdirs to `out_path` if `root_dir` differs with folder containing file.

        :param path: input file path
        :type path: str
        :param root_dir: input root directory path
        :type root_dir: str
        :param out_dir: output directory path
        :type out_dir: str
        """
        assert os.path.exists(path)
        if not os.path.exists(out_dir): os.makedirs(out_dir)
        indir, filename = os.path.split(path)
        subdirs = indir.replace(root_dir, '')

        out_path = os.path.join(out_dir, '%s_%s' %(subdirs.replace('/','_'), filename))
        os.rename(path, out_path)

    # def get_subdir_name(self, path, root_dir):
    #     indir, filename = os.path.split(path)
    #     return indir.replace(root_dir, '')

    def mkdir_and_mkfilepath(self, filename, folds):
        out_dir = ''
        for fold in folds: out_dir = os.path.join(out_dir, fold)
        if not os.path.exists(out_dir): os.makedirs(out_dir)
        return os.path.join(out_dir, filename)

if __name__ =='__main__':
    pass
