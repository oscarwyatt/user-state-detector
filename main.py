# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
# import torch
# from transformers import GPT2Tokenizer
# from trl.gpt2 import GPT2HeadWithValueModel, respond_to_batch
# from trl.ppo import PPOTrainer

def print_hi(name):
    pass
    # Use a breakpoint in the code line below to debug your script.
    # imports
    # get models
    # gpt2_model = GPT2HeadWithValueModel.from_pretrained('gpt2')
    # gpt2_model_ref = GPT2HeadWithValueModel.from_pretrained('gpt2')
    # gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    #
    # # initialize trainer
    # ppo_config = {'batch_size': 1, 'forward_batch_size': 1}
    # ppo_trainer = PPOTrainer(gpt2_model, gpt2_model_ref, **ppo_config)
    #
    # # encode a query
    # query_txt = "This morning I went to the "
    # query_tensor = gpt2_tokenizer.encode(query_txt, return_tensors="pt")
    #
    # # get model response
    # response_tensor = respond_to_batch(gpt2_model, query_tensor, txt_len=1)
    # response_txt = gpt2_tokenizer.decode(response_tensor[0, :])
    # print(response_txt)
    # # define a reward for response
    # # (this could be any reward such as human feedback or output from another model)
    # reward = torch.tensor([1.0])
    #
    # # train model with ppo
    # train_stats = ppo_trainer.step(query_tensor, response_tensor, reward)
    # print(train_stats)

from bert_embeddings import embeddings


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # print(embeddings("why is the grass green"))
    # GovnerEntityPredictor().predict("if you want to get universal credit")
    # print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
