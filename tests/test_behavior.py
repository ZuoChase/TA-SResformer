"""Model-level regression tests for the TA-SResformer core release."""

import hashlib
import json
import unittest

import torch
from timm.models import create_model

import models  # noqa: F401  (registers ta_sresformer with timm)
from models import FRNSTSA, SGCFFN, TASResformer


class ModelCompatibilityTests(unittest.TestCase):
    def test_public_model_names_are_available(self) -> None:
        self.assertTrue(issubclass(TASResformer, torch.nn.Module))
        self.assertTrue(issubclass(FRNSTSA, torch.nn.Module))
        self.assertTrue(issubclass(SGCFFN, torch.nn.Module))

    def test_checkpoint_schema_matches_the_original_release(self) -> None:
        torch.manual_seed(2026)
        model = create_model(
            "ta_sresformer",
            T=4,
            in_channels=2,
            num_classes=18,
            img_size=64,
        )
        keys = list(model.state_dict())
        digest = hashlib.sha256(json.dumps(keys).encode()).hexdigest()

        self.assertEqual(len(keys), 237)
        self.assertEqual(
            digest,
            "f5a89c810b270b7d48f982a6c1147b3ea93a66d72980d39d59108883fbd67bda",
        )
        self.assertEqual(
            sum(parameter.numel() for parameter in model.parameters()),
            2_390_608,
        )


if __name__ == "__main__":
    unittest.main()
