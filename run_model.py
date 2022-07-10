import argparse
import logging
import numpy as np
import pandas as pd
from sklearn import metrics
import transformers
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel, BertConfig
from torch import cuda
from utils import init_logger
import os
import time

logger = logging.getLogger(__name__)


def log_script_arguments(logger, args):
    """
    Log all arguments to the given logger
    :param logger: The logger
    :param args: The arguments given as a namespace - the output of parse_args()
    :return:
    """
    logger.info("Working directory: %s", os.getcwd())
    logger.info("This is the complete list of arguments used:")
    logger.info("===========================================")
    for arg in vars(args):
        arg_val = getattr(args, arg)
        try:
            if isinstance(arg_val, str):
                raise TypeError()
            logger.info("%s: %s" % (arg, ", ".join([str(it) for it in arg_val])))
        except TypeError:
            logger.info("%s: %s" % (arg, arg_val))
    logger.info("===========================================\n")


class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.comment_text = dataframe.comment_text
        self.targets = self.data.list
        self.max_len = max_len

    def __len__(self):
        return len(self.comment_text)

    def __getitem__(self, index):
        comment_text = str(self.comment_text[index])
        comment_text = " ".join(comment_text.split())

        inputs = self.tokenizer.encode_plus(
            comment_text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }


class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained('bert-base-uncased')
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, 7)

    def forward(self, ids, mask, token_type_ids):
        _, output_1 = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output


def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)


def train(epoch, model, training_loader, device, optimizer):
    model.train()
    for _, data in enumerate(training_loader, 0):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.float)

        outputs = model(ids, mask, token_type_ids)

        optimizer.zero_grad()
        loss = loss_fn(outputs, targets)
        if _ % 5000 == 0:
            logger.info(f'{_} \t Epoch: {epoch}, Loss:  {loss.item()}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def validation(model, testing_loader, device):
    model.eval()
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float)
            outputs = model(ids, mask, token_type_ids)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    return fin_outputs, fin_targets


def write_pred_file(outputs, targets):
    pred = []
    for x in outputs:
        y = list(map(int, x))
        pred.append(y)

    pred_df = pd.DataFrame({'pred':pred, 'true':targets})

    pred_df.to_csv(f"./eval/pred_{time.strftime('%Y%m%d-%H%M%S')}.csv")


def run_model(run_model_config):
    init_logger(run_model_config)

    log_script_arguments(logger, run_model_config)

    device = 'cuda' if cuda.is_available() else 'cpu'

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_dataset = pd.read_csv(run_model_config.path_to_input_file, converters={'list': pd.eval})
    test_dataset = pd.read_csv(run_model_config.path_to_test_file, converters={'list': pd.eval})

    logger.info("TRAIN Dataset: {}".format(train_dataset.shape))
    logger.info("TEST Dataset: {}".format(test_dataset.shape))
    logger.info("===========================================")

    training_set = CustomDataset(train_dataset, tokenizer, run_model_config.max_len)
    testing_set = CustomDataset(test_dataset, tokenizer, run_model_config.max_len)

    train_params = {'batch_size': run_model_config.train_batch_size,
                    'shuffle': True,
                    'num_workers': 0
                    }

    test_params = {'batch_size': run_model_config.val_batch_size,
                   'shuffle': True,
                   'num_workers': 0
                   }

    training_loader = DataLoader(training_set, **train_params)
    testing_loader = DataLoader(testing_set, **test_params)

    model = BERTClass()
    model.to(device)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=run_model_config.learning_rate)

    for epoch in range(run_model_config.epochs):
        train(epoch, model, training_loader, device, optimizer)
        logger.info("===========================================")

    # validation
    logger.info("Starting validation")

    outputs, targets = validation(model, testing_loader, device)
    outputs = np.array(outputs) >= 0.5

    write_pred_file(outputs, targets)

    accuracy = metrics.accuracy_score(targets, outputs)
    f1_score_micro = metrics.f1_score(targets, outputs, average='micro')
    f1_score_macro = metrics.f1_score(targets, outputs, average='macro')
    logger.info(f"Accuracy Score = {accuracy}")
    logger.info(f"F1 Score (Micro) = {f1_score_micro}")
    logger.info(f"F1 Score (Macro) = {f1_score_macro}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--path_to_input_file",
        default="./data/train_input.csv",
        type=str,
        help="Path to input file.",
    )

    parser.add_argument(
        "--path_to_test_file",
        default="./data/dev_input.csv",
        type=str,
        help="Path to input file.",
    )

    parser.add_argument(
        "--max_len",
        default=200,
        type=int,
        help="max length.",
    )

    parser.add_argument(
        "--train_batch_size",
        default=32,
        type=int,
        help="Train batch size.",
    )

    parser.add_argument(
        "--val_batch_size",
        default=16,
        type=int,
        help="Validation batch size.",
    )

    parser.add_argument(
        "--epochs",
        default=4,
        type=int,
        help="Number of epochs.",
    )

    parser.add_argument(
        "--learning_rate",
        default=1e-05,
        type=int,
        help="Learning rate.",
    )

    run_model_config = parser.parse_args()
    run_model(run_model_config)
