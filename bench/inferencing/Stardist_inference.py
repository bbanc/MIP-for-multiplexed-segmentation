import torch
from inferencing.utils import preproc_img


def run_inference(img, model):
    img = preproc_img(img)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.permute(0, 3, 1, 2)
    model.to("cpu")
    model.eval()

    with torch.no_grad():
        predictions_ = model(img)
    import torch.nn.functional as F
    
    predictions_["binary_map"] = F.softmax(
                predictions_["nuclei_type_map"], dim=1
            )  # shape: (batch_size, num_nuclei_classes, H, W)
    predictions_["dist_map_sigmoid"] = F.sigmoid(predictions_["dist_map"])
    
    (
        instance_map,
        predictions_["instance_types"],
        _,
    ) = model.calculate_instance_map(
        predictions_["dist_map_sigmoid"],
        predictions_["stardist_map"],
        predictions_["binary_map"],
    )
    
    return instance_map.cpu().numpy()[0,:,:]

def infer_img(img_np, model, label=None,):
    return run_inference(img_np, model)

def infer_stack(stack, model, label=None,):
    return run_inference(stack, model)
