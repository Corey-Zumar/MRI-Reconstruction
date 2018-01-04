import os

from datetime import datetime

TIME_DIR_NAME_FORMAT = "%Y%m%d-%H%M%S"


def _get_output_dir_name(suffix, exp_name=None):
    """
    Parameters
    ------------
    suffix : str
        A suffix to be appended to the directory name
    exp_name : str
        (optional) The name of the experiment. This will
        be appended to the directory name if it is not `None`

    Returns
    ------------   
    str
        The name of the output directory 
    """

    dir_name_base = datetime.now().strftime("%Y%m%d-%H%M%S")
    dir_name = "{base}-{sfx}".format(base=dir_name_base, sfx=suffix)
    if exp_name:
        dir_name = "{dn}-{en}".format(dn=dir_name, en=exp_name)

    return dir_name


def create_output_dir(base_path, suffix, exp_name=None):
    """
    Parameters
    ------------
    base_path : str
        The base directory path under which to create
        the output directory
    suffix : str
        A suffix to be appended to the directory name
    exp_name : str
        (optional) The name of the experiment. This will
        be appended to the directory name if it is not `None`

    Returns
    ------------   
    str
        The name of the created output directory
    """

    dir_name = _get_output_dir_name(suffix=suffix, exp_name=exp_name)
    dir_path = os.path.join(os.path.abspath(base_path), dir_name)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    return dir_path
