import glob
import torch
import os


def saver(model_state_dict, optimizer_state_dict, model_path, epoch, step, max_to_save=8):
    total_models = glob.glob(model_path + '*')
    if len(total_models) >= max_to_save:
        total_models.sort()
        os.remove(total_models[0])

    state_dict = {}
    state_dict["model_state_dict"] = model_state_dict
    state_dict["optimizer_state_dict"] = optimizer_state_dict
    state_dict["step"] = step

    torch.save(state_dict, model_path + '-' + str(epoch))
    print('models {} save successfully!'.format(model_path + '-' + str(epoch)))


def loader(model_path):
    state_dict = torch.load(model_path)
    model_state_dict = state_dict["model_state_dict"]
    optimizer_state_dict = state_dict["optimizer_state_dict"]
    step = state_dict["step"]
    return model_state_dict, optimizer_state_dict, step


def mem_loader(mem_path):
    mem = torch.load(mem_path)
    return mem


def mem_saver(mem_items, path, step, max_to_save=5):
    total_models = glob.glob(path + '*')
    if len(total_models) >= max_to_save:
        total_models.sort()
        os.remove(total_models[0])

    torch.save(mem_items, path + '-' + str(step))
    print('memory {} save successfully!'.format(path + '-' + str(step)))


def only_model_saver(model_state_dict, model_path):
    state_dict = {}
    state_dict["model_state_dict"] = model_state_dict

    torch.save(state_dict, model_path)
    print('models {} save successfully!'.format(model_path))


def only_model_loader(model_path):
    state_dict = torch.load(model_path)
    model_state_dict = state_dict["model_state_dict"]

    return model_state_dict
