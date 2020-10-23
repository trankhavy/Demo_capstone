from mrcnn import utils
from mrcnn import model as modellib
from imgaug import augmenters as iaa

from src.config import DATA_DIR, CTELesionConfig, CONFIGS
from src.dataset import CTEHLesionDataset



def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = CTEHLesionDataset()
    dataset_train.load_lesion(DATA_DIR, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = CTEHLesionDataset()
    dataset_val.load_lesion(DATA_DIR, "validation")
    dataset_val.prepare()


if __name__ == '__main__':

    # Configurations
    config = CTELesionConfig()
    config.display()

    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=CONFIGS.log_dir)

    # Load weights
    print("Loading weights ", weights_path)
    if CONFIGS.pretrain_weight == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    train(model)
