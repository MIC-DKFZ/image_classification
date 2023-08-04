from lightning.pytorch.cli import LightningCLI
from augmentation.policies.base_transform import BaseTransform


class CustomLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        
        parser.link_arguments("trainer.max_epochs", "model.init_args.epochs", apply_on="parse")

        parser.add_argument("--num_classes")
        parser.link_arguments("num_classes", "model.init_args.num_classes", apply_on="parse")

        parser.add_argument("--metrics")
        parser.link_arguments("metrics", "model.init_args.metrics", apply_on="parse")

        #parser.add_class_arguments(get_auto_augmentation_class, "train_transforms", instantiate=True, fail_untyped=False)
        #parser.add_argument("--train_transforms", enable_path=True)#, type=transforms.Compose)
        parser.add_subclass_arguments(BaseTransform, "train_transforms")
        parser.add_subclass_arguments(BaseTransform, "test_transforms")

        parser.link_arguments("train_transforms", "data.init_args.train_transforms", apply_on="parse")
        parser.link_arguments("test_transforms", "data.init_args.test_transforms", apply_on="parse")


        #parser.link_arguments("data.init_args.name", "trainer.logger.init_args.project", apply_on="instantiate")

  