# Will use apple/OpenELM-450M from hugging face and generate answers
from models.model_dpo import AutoDPOModelForCausalLM
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
# the preference data is located in data/M1_preference_data.jsonl

# Load the preference data
import pandas as pd

from tqdm import tqdm
from peft import LoraConfig
from trl import DPOTrainer, DPOConfig
from datasets import Dataset
import os


from sklearn.model_selection import train_test_split
login(token="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")
policy_model_name ="BaptisteL/oelm_dpo_270m_fd"
policy_tokenizer_name = "BaptisteL/oelm_dpo_270m_fd"
policy_model = AutoModelForCausalLM.from_pretrained(policy_model_name, trust_remote_code=True, device_map="auto")
policy_tokenizer = AutoTokenizer.from_pretrained(policy_tokenizer_name, trust_remote_code=True, device_map="auto")

reference_model_name ="BaptisteL/oelm_dpo_270m_fd"
reference_tokenizer_name = "BaptisteL/oelm_dpo_270m_fd"
reference_model = AutoModelForCausalLM.from_pretrained(reference_model_name, trust_remote_code=True, device_map="auto")
reference_tokenizer = AutoTokenizer.from_pretrained(reference_tokenizer_name, trust_remote_code=True, device_map="auto")

qs1 = "Question: There are N philosphers sitting around a circular table eating spaghetti and discussing philosphy. The problem is that each philosopher needs two forks to eat, and there are only $N$ forks, one between each pair of philosophers. We want to design an algorithm that the philosophers can use, that ensures that no one starves as long as each philosopher eventually stops eating, and such that the maximum number of philosophers can eat at once. Lecture 5 provides one possible solution which uses a central arbiter. Can you write the philospherTurn function without a central arbiter? You may modify the provided class Fork if required.  class Fork() {   var inUse: Boolean = false  }  def philosopherTurn(l: Fork, r: Fork): Boolean = ??? // your implementation here // your implementation here  def run() =     val n = 5     val forks = new Array[Fork](n)     val philosophers = new Array[Thread](n)     for p <- 0 to n - 1 do         forks(p) = new Fork()      for p <- 0 to n - 1 do         philosophers(p) = new Thread {             override def run() = {                 while (!philosopherTurn(forks(p % n), forks((p + 1) % n))) { /* wait */ }             }         }         philosophers(p).start      for p <- 0 to n - 1 do         philosophers(p).join() Hint: Use the deadlock prevention technique introduced in the lecture."
qs2 = "Question: Consider the loss function $L: \\R^d \to \\R$, $L(\\wv) = \frac{\beta}{2}\\|\\wv\\|^2$, where $\beta > 0$ is a constant. We run gradient descent on $L$ with a stepsize $\\gamma > 0$ starting from some $\\wv_0 \neq 0$. Which of the statements below is true? ?\n\nOptions:\nA. Gradient descent converges to the global minimum for any stepsize $\\gamma > 0$.\nB. Gradient descent with stepsize $\\gamma = \frac{2}{\beta}$ produces iterates that diverge to infinity ($\\|\\wv_t\\| \to \\infty$ as $t\to \\infty$).\nC. Gradient descent converges in two steps for $\\gamma = \frac{1}{\beta}$ (i.e., $\\wv_2$ is the \textbf{first} iterate attaining the global minimum of $L$).\nD. Gradient descent converges to the global minimum for any stepsize in the interval $\\gamma \\in \big( 0, \frac{2}{\beta}\big)$.\n\n Answer:"

p_encoded_qs1 = policy_tokenizer(qs1, return_tensors="pt", padding="max_length", max_length=1024, truncation=True, )
p_encoded_qs2 = policy_tokenizer(qs2, return_tensors="pt", padding="max_length", max_length=1024, truncation=True, )


r_encoded_qs1 = reference_tokenizer(qs1, return_tensors="pt", padding="max_length", max_length=1024, truncation=True, )
r_encoded_qs2 = reference_tokenizer(qs2, return_tensors="pt", padding="max_length", max_length=1024, truncation=True, )

p_output_qs1 = policy_model.generate(**p_encoded_qs1, max_new_tokens=1024, eos_token_id = policy_tokenizer.eos_token_id, pad_token_id = policy_tokenizer.eos_token_id)
p_output_qs2 = policy_model.generate(**p_encoded_qs2, max_new_tokens=1, eos_token_id = policy_tokenizer.eos_token_id, pad_token_id = policy_tokenizer.eos_token_id)


r_output_qs1 = reference_model.generate(**r_encoded_qs1, max_new_tokens=1024, eos_token_id = reference_tokenizer.eos_token_id, pad_token_id = reference_tokenizer.eos_token_id)
r_output_qs2 = reference_model.generate(**r_encoded_qs2, max_new_tokens=1, eos_token_id = reference_tokenizer.eos_token_id, pad_token_id = reference_tokenizer.eos_token_id)

p_output_qs1_text = policy_tokenizer.decode(p_output_qs1[0], skip_special_tokens=True)
p_output_qs2_text = policy_tokenizer.decode(p_output_qs2[0], skip_special_tokens=True)

r_output_qs1_text = reference_tokenizer.decode(r_output_qs1[0], skip_special_tokens=True)
r_output_qs2_text = reference_tokenizer.decode(r_output_qs2[0], skip_special_tokens=True)

print('Qs1:', qs1)
print('Policy:', p_output_qs1_text)
print('Reference:', r_output_qs1_text)
print("\n\n")
print('Qs2:', qs2)
print('Policy:', p_output_qs2_text)
print('Reference:', r_output_qs2_text)
