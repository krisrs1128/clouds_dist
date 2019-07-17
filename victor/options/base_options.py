class BaseOptions:
    def __init__(self, parser):
        self.parser = parser
        parser.add_argument(
            "--n_epochs", type=int, default=200, help="number of epochs of training"
        )
        parser.add_argument(
            "--batch_size", type=int, default=64, help="size of the batches"
        )
        parser.add_argument(
            "--lr", type=float, default=0.0002, help="adam: learning rate"
        )
        parser.add_argument(
            "--b1",
            type=float,
            default=0.5,
            help="adam: decay of first order momentum of gradient",
        )
        parser.add_argument(
            "--b2",
            type=float,
            default=0.999,
            help="adam: decay of first order momentum of gradient",
        )
        parser.add_argument(
            "--n_cpu",
            type=int,
            default=8,
            help="number of cpu threads to use during batch generation",
        )
        parser.add_argument(
            "--latent_dim",
            type=int,
            default=100,
            help="dimensionality of the latent space",
        )
        parser.add_argument(
            "--n_classes", type=int, default=10, help="number of classes for dataset"
        )
        parser.add_argument(
            "--img_size", type=int, default=32, help="size of each image dimension"
        )
        parser.add_argument(
            "--channels", type=int, default=1, help="number of image channels"
        )
        parser.add_argument(
            "--sample_interval",
            type=int,
            default=400,
            help="interval between image sampling",
        )
        parser.add_argument(
            "--source", type=str, default="./data", help="Where to find the data"
        )
        parser.add_argument(
            "--dest",
            type=str,
            default="./results",
            help="Where to store the models and inferences",
        )

    def parse(self):
        self.opt = self.parser.parse_args()
        return self.opt
