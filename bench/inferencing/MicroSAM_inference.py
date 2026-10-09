from micro_sam.automatic_segmentation import automatic_instance_segmentation


def infer_img_zero_shot(img, model, **generate_kwargs):
    return automatic_instance_segmentation(
        model.predictor, model.segmenter, input_path=img, ndim=2, verbose=False, **generate_kwargs
    )


def infer_stack_zero_shot(stack, model, **generate_kwargs):
    return infer_img_zero_shot(stack, model, **generate_kwargs)
