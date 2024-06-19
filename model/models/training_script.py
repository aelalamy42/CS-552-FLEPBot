from models.model_dpo import AutoDPOModelForCausalLM
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

from transformers import get_polynomial_decay_schedule_with_warmup
from typing import Tuple

# the preference data is located in data/M1_preference_data.jsonl

# Load the preference data
import pandas as pd
import wandb

from peft import LoraConfig
from trl import DPOTrainer, DPOConfig
from datasets import Dataset
import sys
import torch.nn.functional as F

from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    #get the arguments from the command line
    wandb.login(key="", relogin=True)
    wandb.init(
            project="MNLP-M3",
            name = sys.argv[6]
            )
    # parsing the training arguments
    batch_size = int(sys.argv[2])
    training_dataset = sys.argv[4]
    output_dir = sys.argv[6]
    model_name = sys.argv[8]
    tokenizer_name = model_name

    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    preference_data = pd.read_json(training_dataset, lines=True)
    train_data, eval_data = train_test_split(preference_data, test_size=0.1)


    training_args = DPOConfig(
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        save_steps=500,
 #       learning_rate=1e-4,
        logging_steps=50,
        output_dir='./MNLP/output/',
        report_to="wandb",
        run_name="dpo_oelm_270m_scratch",
        max_prompt_length=1000,
        max_length=1000,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    lr_scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=500,
            num_training_steps=10000,
            lr_end=5e-6,
    )
    # 5. initialize the DPO trainer
    dpo_trainer = DPOTrainer(
        model,
    	optimizers = (optimizer, lr_scheduler),
        args=training_args,
        train_dataset=Dataset.from_pandas(train_data),
        eval_dataset=Dataset.from_pandas(eval_data),
        tokenizer=tokenizer,
    )

    def dpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the DPO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the DPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        """
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        pi_logratios = pi_logratios.to(self.accelerator.device)
        ref_logratios = ref_logratios

        # Compute the length of the chosen and reference responses
        chosen_length = policy_chosen_logps.size(0)
        reference_length = reference_chosen_logps.size(0)

        # Before: logits = pi_logratios - ref_logratios
        logits = self.beta * (pi_logratios - ref_logratios) - self.alpha * (chosen_length - reference_length)

        losses = (
            -F.logsigmoid(logits) * (1 - self.label_smoothing)
            - F.logsigmoid(-logits) * self.label_smoothing
        )

        chosen_rewards = (
            self.beta
            * (
                policy_chosen_logps.to(self.accelerator.device) - reference_chosen_logps.to(self.accelerator.device)
            ).detach()
        )
        rejected_rewards = (
            self.beta
            * (
                policy_rejected_logps.to(self.accelerator.device)
                - reference_rejected_logps.to(self.accelerator.device)
            ).detach()
        )

        return losses, chosen_rewards, rejected_rewards
    
    function_type = type(dpo_trainer.dpo_loss)
    dpo_trainer.dpo_loss = function_type(dpo_loss, dpo_trainer)
    
    dpo_trainer.alpha = 0.1

    # 6. train the model
    dpo_trainer.train()
    model.save_pretrained(output_dir+'_m')
    tokenizer.save_pretrained(output_dir+'_m')
    dpo_trainer.model.save_pretrained(output_dir+'_m')
    wandb.finish()
