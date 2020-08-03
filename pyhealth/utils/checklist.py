import os

def check_expdata_dir(expdata_id):
    """
    Check whether the exp data folder  exist,
        If not, will create the folder

    Parameters

    ----------
    root_dir :  str,
        root dir of current project

    expdata_id : str, optional (default='init.test') 
        name of current experiment data

    """

    r_root = os.path.join('./', 'experiments_data')
    if os.path.exists(r_root) is False:
        os.mkdir(r_root)
    exp_root = os.path.join(r_root, expdata_id)
    if os.path.exists(exp_root) is False:
        os.mkdir(exp_root)

def check_model_dir(expmodel_id):
    """
    Check whether the checkouts/results folders of current experiment(exp_id) exist,
        If not, will create both folders

    Parameters

    ----------
    root_dir :  str,
        root dir of current project

    expmodel_id : str, optional (default='init.test') 
        name of current experiment

    """

    r_root = os.path.join('./', 'experiments_records')
    if os.path.exists(r_root) is False:
        os.mkdir(r_root)
    exp_root = os.path.join(r_root, expmodel_id)
    if os.path.exists(exp_root) is False:
        os.mkdir(exp_root)
    checkout_dir = os.path.join(exp_root, 'checkouts')
    result_dir = os.path.join(exp_root, 'results')
    if os.path.exists(checkout_dir) is False:
        os.mkdir(checkout_dir)
    if os.path.exists(result_dir) is False:
        os.mkdir(result_dir)
