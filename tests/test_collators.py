import torch

from data.collators import DataCollatorForSupervisedDataset


class _TokenizerWithSharedPadAndEos:
    pad_token_id = 2
    eos_token_id = 2


def test_supervised_collator_preserves_real_eos_attention():
    collator = DataCollatorForSupervisedDataset(_TokenizerWithSharedPadAndEos())
    instances = [
        {
            "input_ids": torch.tensor([1, 2, 3]),
            "attention_mask": torch.tensor([1, 1, 1]),
            "labels": torch.tensor([-100, -100, 3]),
        },
        {
            "input_ids": torch.tensor([1, 4]),
            "attention_mask": torch.tensor([1, 1]),
            "labels": torch.tensor([-100, 4]),
        },
    ]

    batch = collator(instances)

    assert torch.equal(batch["input_ids"], torch.tensor([[1, 2, 3], [1, 4, 2]]))
    assert torch.equal(batch["attention_mask"], torch.tensor([[1, 1, 1], [1, 1, 0]]))
