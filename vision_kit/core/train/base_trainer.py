from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torchinfo import summary
from vision_kit.models.architectures import build_model
from vision_kit.utils.logging_utils import logger
from vision_kit.utils.model_utils import load_ckpt, process_ckpts


class TrainingModule(pl.LightningModule):
    def __init__(self, cfg, evaluator=None, pretrained: bool = True) -> None:
        super(TrainingModule, self).__init__()

        self.save_hyperparameters(ignore=["evaluator"])
        self.example_input_array = torch.ones((1, 3, *(cfg.model.input_size)))

        self.ema_model = None
        self.model = None
        self.evaluator = evaluator
        self.pretrained = pretrained

    def model_info(self):
        model_info = summary(self.model, input_size=self.example_input_array.shape, verbose=0, depth=2)
        logger.info(f"Model Info\n{model_info}")

    def training_step(self, *args, **kwargs) -> STEP_OUTPUT:
        return super().training_step(*args, **kwargs)

    def validation_step(self, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        return super().validation_step(*args, **kwargs)

    def test_step(self, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        return super().test_step(*args, **kwargs)

    def configure_optimizers(self):
        return super().configure_optimizers()

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint["model"] = self.get_model(half=True).state_dict()

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        model_weight, ema_weight = process_ckpts(checkpoint)
        if self.ema_model:
            if len(ema_weight) != len(self.model.state_dict()):
                logger.info(
                    "Weight averaging is enabled but average state does not"
                    "match the model"
                )
            else:
                # self.ema_model.module.load_state_dict(ema_weight)
                self.ema_model.updates = checkpoint["epoch"]
                logger.info("Loaded average state from checkpoint.")
        else:
            checkpoint["state_dict"] = model_weight

    @torch.no_grad()
    def to_torchscript(self, file_path: Optional[Union[str, Path]] = None, method: Optional[str] = "script",
                       example_inputs: Optional[Any] = None, **kwargs):
        script_model = self.get_model()

        script_model.head.export = True
        logger.info("Fusing Layers...")
        script_model.fuse()
        script_model.eval()

        logger.info(f'Starting export with torch {torch.__version__}...')
        if method == "script":
            ts = torch.jit.script(script_model, **kwargs)
        elif method == "trace":
            # if no example inputs are provided, try to see if model has example_input_array set
            if example_inputs is None:
                if self.example_input_array is None:
                    raise ValueError(
                        "Choosing method=`trace` requires either `example_inputs`"
                        " or `model.example_input_array` to be defined."
                    )
                example_inputs: torch.Tensor = self.example_input_array

            example_inputs = example_inputs.to(self.device)
            script_model = script_model.to(self.device)
            ts = torch.jit.trace(script_model, example_inputs=example_inputs, **kwargs)
        else:
            raise ValueError(f"The 'method' parameter only supports 'script' or 'trace', but value given was: {method}")

        ts.save(file_path)
        logger.info(f'Saved torchscript model at: {file_path}')

    @torch.no_grad()
    def to_onnx(
            self, file_path: Union[str, Path],
            input_sample: Optional[Any] = None, simplify: bool = True, **kwargs):
        import onnx

        model = self.get_model()
        model.head.export = True
        logger.info("Fusing Layers...")
        model.fuse()
        model.eval()

        if input_sample is None:
            if self.example_input_array is None:
                raise ValueError(
                    "Onnx conversion requires either `input_sample`"
                    " or `model.example_input_array` to be defined."
                )
            input_sample: torch.Tensor = self.example_input_array
        input_sample = input_sample.cpu()
        model = model.cpu()

        torch.onnx.export(
            model,
            input_sample,
            file_path,
            **kwargs
        )
        # Checks
        model_onnx = onnx.load(file_path)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model

        # Simplify
        if simplify:
            try:
                import onnxsim

                logger.info(f'Simplifying with onnx-simplifier {onnxsim.__version__}...')
                model_onnx, check = onnxsim.simplify(model_onnx)
                assert check, 'assert check failed'
                onnx.save(model_onnx, file_path)
            except Exception as e:
                logger.info(f'Simplifier failure: {e}')
        logger.info(f'Saved onnx model at: {file_path}')

    def get_model(self, half: bool = False):
        if self.ema_model:
            model = deepcopy(self.ema_model.module)
        else:
            model = deepcopy(self.model)
        if half:
            model.half()
        return model

    @staticmethod
    def load_pretrained(cfg):
        model = build_model(cfg)
        state_dict = torch.load(cfg.model.weight, map_location="cpu")
        model = load_ckpt(model, state_dict)
        return model
