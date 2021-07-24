from .posenet.PoseNet import PoseNet
from .transposenet.EMSTransPoseNet import EMSTransPoseNet
from  .transposenet.MSTransPoseNet import MSTransPoseNet

def get_model(model_name, backbone_path, config):
    """
    Get the instance of the request model
    :param model_name: (str) model name
    :param backbone_path: (str) path to a .pth backbone
    :param config: (dict) config file
    :return: instance of the model (nn.Module)
    """
    if model_name == 'posenet':
        return PoseNet(backbone_path)
    elif model_name == 'ms-transposenet':
        return MSTransPoseNet(config, backbone_path)
    elif model_name == 'ems-transposenet':
        return EMSTransPoseNet(config, backbone_path)
    else:
        raise "{} not supported".format(model_name)