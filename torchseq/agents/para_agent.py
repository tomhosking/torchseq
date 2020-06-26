import torch
import torch.nn.functional as F
from torch import nn

from torchseq.agents.model_agent import ModelAgent
from torchseq.datasets.paraphrase_dataset import ParaphraseDataset
from torchseq.datasets.paraphrase_loader import ParaphraseDataLoader
from torchseq.datasets.paraphrase_pair import ParaphrasePair
from torchseq.datasets.preprocessed_loader import PreprocessedDataLoader
from torchseq.datasets.squad_dataset import SquadDataset
from torchseq.datasets.squad_loader import SquadDataLoader
from torchseq.models.para_transformer import TransformerParaphraseModel
from torchseq.models.pretrained_modular import PretrainedModularModel
from torchseq.models.suppression_loss import SuppressionLoss
from torchseq.models.kl_divergence import get_kl
from torchseq.utils.tokenizer import Tokenizer


class ParaphraseAgent(ModelAgent):
    def __init__(self, config, run_id, output_path, silent=False, training_mode=True):
        super().__init__(config, run_id, output_path, silent, training_mode)

        self.tgt_field = "s1" if self.config.training.data.get("flip_pairs", False) else "s2"

        # define data_loader
        if self.config.training.use_preprocessed_data:
            self.data_loader = PreprocessedDataLoader(config=config)
        else:
            if (
                self.config.training.dataset
                in [
                    "paranmt",
                    "parabank",
                    "kaggle",
                    "parabank-qs",
                    "para-squad",
                    "models/squad-udep",
                    "models/squad-constituency",
                    "models/squad-udep-deptree",
                    "models/qdmr-squad",
                    "models/nq_newsqa-udep",
                    "models/nq_newsqa-udep-deptree",
                    "models/squad_nq_newsqa-udep",
                    "models/squad_nq_newsqa-udep-deptree",
                    "models/naturalquestions-udep",
                    "models/newsqa-udep",
                    "models/naturalquestions-udep-deptree",
                    "models/newsqa-udep-deptree",
                ]
                or self.config.training.dataset[:5] == "qdmr-"
                or "kaggle-" in self.config.training.dataset
            ):
                self.data_loader = ParaphraseDataLoader(config=config)
                self.src_field = (
                    "s2"
                    if (self.config.task == "autoencoder" or self.config.training.data.get("flip_pairs", False))
                    else "s1"
                )
            elif self.config.training.dataset in [
                "squad",
                "squad_nq_newsqa",
                "nq_newsqa",
                "newsqa",
                "naturalquestions",
            ]:
                self.data_loader = SquadDataLoader(config=config)
                self.src_field = "q"
                self.tgt_field = "q"
            else:
                raise Exception("Unrecognised dataset: {:}".format(config.training.dataset))

        # define loss
        self.loss = nn.CrossEntropyLoss(ignore_index=Tokenizer().pad_id, reduction="none")

        # define model
        if self.config.data.get("model", None) is not None and self.config.model == "pretrained_modular":
            self.model = PretrainedModularModel(self.config, src_field=self.src_field, loss=self.loss)
        else:
            self.model = TransformerParaphraseModel(self.config, src_field=self.src_field, loss=self.loss)

        self.suppression_loss = SuppressionLoss(self.config)

        # define optimizer
        self.create_optimizer()

        self.set_device()

        self.create_samplers()

    def step_train(self, batch, tgt_field):
        loss = 0

        output, logits, _, memory = self.decode_teacher_force(self.model, batch, tgt_field)

        this_loss = self.loss(logits.permute(0, 2, 1), batch[tgt_field])

        if self.config.training.suppression_loss_weight > 0:
            this_loss += self.config.training.suppression_loss_weight * self.suppression_loss(
                logits, batch[self.src_field]
            )

        this_loss = torch.sum(this_loss, dim=1) / (batch[tgt_field + "_len"] - 1).to(this_loss)

        loss += torch.mean(this_loss, dim=0)

        if self.config.encdec.data.get("variational", False) or self.config.data.get("variational_projection", False):
            kl_loss = torch.mean(get_kl(memory["mu"], memory["logvar"]))

            kl_warmup_steps = self.config.training.data.get("kl_warmup_steps", 0)

            kl_weight = (
                1
                if self.global_step >= 2 * kl_warmup_steps
                else (
                    0
                    if self.global_step < kl_warmup_steps
                    else float(self.global_step - kl_warmup_steps) / (1.0 * kl_warmup_steps)
                )
            )

            loss += kl_loss * kl_weight

        return loss

    def text_to_batch(self, x, device):
        if self.config.training.dataset in ["squad"]:
            x["s2"] = ""

            return {
                k: (v.to(self.device) if k[-5:] != "_text" else v)
                for k, v in SquadDataset.pad_and_order_sequences(
                    [SquadDataset.to_tensor(x, tok_window=self.config.prepro.tok_window)]
                ).items()
            }
        else:
            x["s2"] = ""

            return {
                k: (v.to(self.device) if k[-5:] != "_text" else v)
                for k, v in ParaphraseDataset.pad_and_order_sequences(
                    [ParaphraseDataset.to_tensor(x, tok_window=self.config.prepro.tok_window)]
                ).items()
            }
