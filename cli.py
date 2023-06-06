from lightning.pytorch.cli import LightningCLI


class CustomLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("data.init_args.dataset", "trainer.logger.init_args.name", apply_on="instantiate")

        parser.add_argument("--num_classes")
        parser.link_arguments("num_classes", "model.init_args.num_classes", apply_on="parse")