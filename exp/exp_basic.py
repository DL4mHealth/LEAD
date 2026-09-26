import importlib
import os

import torch


class Exp_Basic(object):
    """Base experiment with lazy model imports.

    Baseline models have optional third-party dependencies. Import only the
    selected model so a missing dependency of an unused baseline cannot prevent
    LEAD experiments from starting.
    """

    AVAILABLE_MODELS = {
        'ADformer', 'TCN', 'Transformer', 'EEGConformer', 'TimesNet', 'MedGNN',
        'CBraMod', 'EEGNet', 'ManualFeature', 'EEGInception', 'BIOT', 'PatchTST',
        'ModernTCN', 'LaBraM', 'MNet', 'iTransformer', 'CSBrain', 'LEAD', 'EEGDeformer', 'REVE',
    }

    def __init__(self, args):
        self.args = args
        if args.model not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Unknown model '{args.model}'. Available models: {sorted(self.AVAILABLE_MODELS)}"
            )
        try:
            selected_model = importlib.import_module(f"models.{args.model}")
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                f"Could not import model '{args.model}'. A model-specific optional dependency "
                f"may be missing. Original error: {exc}"
            ) from exc
        self.model_dict = {args.model: selected_model}
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = (
                str(self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            )
            device = torch.device(f'cuda:{self.args.gpu}')
            print(f'Use GPU: cuda:{self.args.gpu}')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _print_common_run_config(self):
        """Print configuration shared by pretrain and downstream experiments."""
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total parameters: {total_params}")
        print(f"Utilized sampling rate(s): {getattr(self.args, 'sampling_rate_list', 'all')}")

        # These embedding choices are specific to LEAD. Baselines still print the
        # utilized sampling-rate filter above, but should not imply that they use
        # LEAD's embedding modules.
        if getattr(self.args, "model", None) == "LEAD":
            print(
                f"Sampling-rate embedding: "
                f"{bool(getattr(self.args, 'use_sampling_embedding', False))}"
            )
            print(
                f"Temporal positional embedding: "
                f"{getattr(self.args, 'temporal_pos_type', 'learnable')}"
            )
            print(
                f"Channel positional embedding: "
                f"{getattr(self.args, 'channel_pos_type', '3D')}"
            )

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
