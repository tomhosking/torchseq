from torchseq.datasets.builder import dataloader_from_config


def generate(instance, samples=[]):
    data_loader = dataloader_from_config(config=instance.config, data_path=instance.data_path, dev_samples=samples)
    _, _, (pred_output, _, _), _ = instance.inference(data_loader.valid_loader, desc="Generating")

    return pred_output
