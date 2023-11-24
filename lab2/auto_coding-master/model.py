import json
import logging
import os
from typing import Dict

import torch
from torch import nn, Tensor
from transformers import GPT2LMHeadModel, GPT2Tokenizer

logging.basicConfig(
    format=logging.BASIC_FORMAT,
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class GPTSingleHead(nn.Module):
    """
    Different from directly using GPT2LMHeadModel, this wraps up GPT2LMHeadModel as well as GPT2Tokenizer
    """

    def __init__(self, model_name_or_path: str, max_seq_length: int = 256, do_lower_case: bool = False,
                 special_words_to_add=None):
        """
        定义了一个名为 GPTSingleHead 的 PyTorch 模型类，用于创建 GPT2 模型
        :param model_name_or_path: 指定要加载或初始化的 GPT2 模型的名称或路径。
        :param max_seq_length: 指定输入序列的最大长度。
        :param do_lower_case: 指定是否将输入文本转换为小写。
        :param special_words_to_add: 一个可选参数，用于指定要添加到 tokenizer 中的特殊词语。如 <python>, <java>
        """
        super(GPTSingleHead, self).__init__()
        self.config_keys = ['max_seq_length', 'do_lower_case']
        self.do_lower_case = do_lower_case
        if max_seq_length > 1024:
            logging.warning(
                "GPT only allows a max_seq_length of 1024. Value will be set to 1024")
            max_seq_length = 1024
        self.max_seq_length = max_seq_length
        self.gpt = GPT2LMHeadModel.from_pretrained(model_name_or_path)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path, do_lower_case=do_lower_case)
        if special_words_to_add != None:
            self.add_special_words(special_words_to_add)

        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        # self.pad_token_id=self.tokenizer.pad_token_id

    def tokenize(self, text: str):  # default for cls
        """
        将输入文本转换为 token IDs 的序列。
        首先使用 tokenizer.tokenize 将文本标记化为 token 列表，然后使用 tokenizer.convert_tokens_to_ids 将 token 列表转换为对应的 token IDs。
        :param text: 输入文本
        :return: token IDs 的序列
        """
        return self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))

    def add_special_words(self, special_words_to_add):
        """
        添加特殊词语到 tokenizer 中，并调整模型的 token embeddings 大小以适应新的词汇量。
        :param special_words_to_add: 要添加到 tokenizer 中的特殊词语
        :return: None
        """
        orig_num_tokens = len(self.tokenizer)
        num_added_tokens = self.tokenizer.add_special_tokens(special_words_to_add)
        if num_added_tokens > 0:
            self.gpt.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens)

    def forward(self, input: Dict[str, torch.Tensor]):
        """
        定义模型的前向传播逻辑。接收一个名为 input 的字典作为输入，包含键为"input_ids"的输入token IDs。
        使用 self.gpt 模型将 input["input_ids"]作 为输入，并返回损失和 logit（模型的输出）。

        :param input: 一个字典，包含键为"input_ids"的输入token IDs。
        :return:  损失和 logit（模型的输出）
        """
        loss, logits = self.gpt(input["input_ids"], labels=input["input_ids"])[:2]
        return loss, logits

    def get_config_dict(self):
        """
        返回模型的配置字典，该字典包含在初始化函数中定义的配置参数。
        """
        return {key: self.__dict__[key] for key in self.config_keys}

    def padding_features(self, features_dict_list):
        """
        padding features for a batch
        对一个batch的特征进行padding。
        遍历features_dict_list中的每个特征字典，将每个特征的token IDs加入到对应的batch列表中。
        找到batch中最长的输入序列长度max_input_len_this_batch。
        根据max_input_len_this_batch对每个特征的token IDs进行padding，使其长度一致。
        :param features_dict_list: i.e., batch
        :return: padded batch features
        """
        max_input_len_this_batch = 0

        batch_features = {feature_name: [] for feature_name in features_dict_list[0]}
        for feature_dict in features_dict_list:
            for feature_name, feature_ids in feature_dict.items():
                if feature_name == "input_ids" and len(feature_ids) > max_input_len_this_batch:
                    max_input_len_this_batch = len(feature_ids)
                batch_features[feature_name].append(feature_ids)

        padded_batch_features = {feature_name: [] for feature_name in features_dict_list[0]}
        for feature_name, batch_ids in batch_features.items():

            for each_ids in batch_ids:
                padded = each_ids + [self.tokenizer.pad_token_id] * (max_input_len_this_batch - len(each_ids))
                padded_batch_features[feature_name].append(padded)

        for feature_name, ids in padded_batch_features.items():
            padded_batch_features[feature_name] = torch.tensor(ids)

        return padded_batch_features

    def get_embedding_dimension(self) -> int:
        """
        返回模型的嵌入维度。
        """
        return self.gpt.config.hidden_size

    def get_config(self) -> int:
        """
        返回模型的配置。
        """
        return self.gpt.config

    def save(self, output_path: str):
        """
        保存模型的权重、tokenizer和配置字典到指定路径。
        :param output_path: 模型保存路径
        :return: None
        """
        self.gpt.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        with open(os.path.join(output_path, 'gpt_sh_config.json'), 'w') as f:
            json.dump(self.get_config_dict(), f, indent=2)

    def reload(self, input_path: str):
        """reload from checkpoint weights"""
        return GPTSingleHead.load(input_path + "/0_GPTSingleHead")

    @staticmethod
    def load(input_path: str):
        if not os.path.isfile(os.path.join(input_path, 'gpt_sh_config.json')):
            raise ValueError("In the model path does not find gpt_sh_config.json file, you may have not trained yet")
        with open(os.path.join(input_path, 'gpt_sh_config.json')) as f:
            config = json.load(f)
        return GPTSingleHead(model_name_or_path=input_path, **config)


class EmptyHeads(nn.Module):
    def __init__(self):
        self.config_keys = []
        super().__init__()

    def forward(self, input: Dict[str, Tensor]):
        return input

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path):
        with open(os.path.join(output_path, 'empty_heads_config.json'), 'w') as f:
            json.dump(self.get_config_dict(), f, indent=2)
        torch.save(self.state_dict(), os.path.join(output_path, 'empty_heads.pt'))

    def load_saved(self, input_path):
        self.load_state_dict(torch.load(os.path.join(input_path, '1_EmptyHeads', 'empty_heads.pt')))

    @staticmethod
    def load(input_path, config):
        if not os.path.isfile(os.path.join(input_path, 'empty_heads_config.json')):
            raise ValueError(
                "In the model path does not find empty_heads_config.json file, you may have not trained yet")

        with open(os.path.join(input_path, 'empty_heads_config.json')) as f:
            config = json.load(f)
        model = EmptyHeads()

        if not os.path.isfile(os.path.join(input_path, 'empty_heads.pt')):
            raise ValueError("In the model path does not find state of file, you need to train and get weights first")

        model.load_state_dict(torch.load(os.path.join(input_path, 'empty_heads.pt')))
        return model
