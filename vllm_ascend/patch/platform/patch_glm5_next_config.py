# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Patch target: vllm/transformers_utils/model_arch_config_convertor.py
# - Add glm5_next to the is_deepseek_mla model type list so that the model
#   is recognized as an MLA model and uses the correct attention backend.

from vllm.transformers_utils.model_arch_config_convertor import (
    ModelArchConfigConvertorBase,
)

_original_is_deepseek_mla = ModelArchConfigConvertorBase.is_deepseek_mla


def _patched_is_deepseek_mla(self) -> bool:
    if (hasattr(self, 'hf_text_config')
        and getattr(self.hf_text_config, 'model_type', None) == 'glm5_next'):
        # glm5_next uses MLA (kv_lora_rank != None) and DSA indexer
        return getattr(self.hf_text_config, 'kv_lora_rank', None) is not None
    return _original_is_deepseek_mla(self)


ModelArchConfigConvertorBase.is_deepseek_mla = _patched_is_deepseek_mla
