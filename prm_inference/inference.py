from transformers import AutoTokenizer  # 从transformers库导入AutoTokenizer
from transformers import AutoModelForCausalLM  # 从transformers库导入AutoModelForCausalLM
import torch  # 导入torch库

good_token = '+'  # 定义正面标记
bad_token = '-'  # 定义负面标记
step_tag = 'ки'  # 定义步骤标记

# 从预训练模型加载分词器
tokenizer = AutoTokenizer.from_pretrained('F:/code/github/math-feedback/math-feedback/prm_inference/models/peiyi9979_math_shepherd_mistral_7b_prm')

# tokenizer.encode用于将文本转换为模型可以理解的格式，即将文本转换为一系列的 token ID，返回的是一个整数列表，代表了输入文本的 token ID 序列。
# 对定义的正面和负面标记进行编码，并取结果列表中的第二个元素及之后的元素
candidate_tokens = tokenizer.encode(f"{good_token} {bad_token}")[1:]  # [648, 387]
# 这是因为在某些 tokenizer 中，第一个 token 通常是一个特殊的开始标记（如 [CLS] 或 <s>），并不代表实际的文本内容。所以这里通过 [1:] 去掉了第一个 token。

# 对步骤标记进行编码，并取结果列表中的最后一个元素
step_tag_id = tokenizer.encode(f"{step_tag}")[-1]  # 12902
# 因为步骤标记在编码后的最后一个 token 是我们关心的部分，或者说步骤标记只有最后一个是有效的token，所以直接取最后一个元素。

# 从预训练模型加载并设置为评估模式
model = AutoModelForCausalLM.from_pretrained('F:/code/github/math-feedback/math-feedback/prm_inference/models/peiyi9979_math_shepherd_mistral_7b_prm').eval()
# 在 PyTorch 中，.eval() 是一个模式切换方法，用于将模型切换到评估模式（evaluation mode）。这是因为在训练模式和评估模式下，某些层的行为是不同的。
# 例如，Dropout 层在训练模式下会随机丢弃一些神经元以防止过拟合，但在评估模式下，我们需要使用所有的神经元来得到最准确的预测。另一个例子是 Batch Normalization 层，它在训练模式下会根据当前 batch 的数据进行归一化，但在评估模式下，它会使用在训练过程中累积的运行平均值和方差进行归一化。
# 因此，当需要使用模型进行预测或评估时应该先调用 .eval() 将模型切换到评估模式。这样可以确保模型在预测时的行为是正确的。

# 定义问题
question = """Janet\u2019s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"""
# 定义输出1，正确答案
output1 = """Step 1: Janet's ducks lay 16 eggs per day. ки\nStep 2: She eats three for breakfast every morning, so she has 16 - 3 = 13 eggs left. ки\nStep 3: She bakes muffins for her friends every day with four eggs, so she has 13 - 4 = 9 eggs left. ки\nStep 4: She sells the remainder at the farmers' market daily for $2 per fresh duck egg, so she makes 9 * $2 = $18 every day at the farmers' market. The answer is: 18 ки""" # 18 is right
# 定义输出2，错误答案
output2 = """Step 1: Janet's ducks lay 16 eggs per day. ки\nStep 2: She eats three for breakfast every morning, so she has 16 - 3 = 13 eggs left. ки\nStep 3: She bakes muffins for her friends every day with four eggs, so she has 13 - 4 = 9 eggs left. ки\nStep 4: She sells the remainder at the farmers' market daily for $2 per fresh duck egg, so she makes 9 * $2 = $17 every day at the farmers' market. The answer is: 17 ки""" # 17 is wrong

# 对两个输出进行处理
for output in [output1, output2]:
    input_for_prm = f"{question} {output}"  # 将问题和输出合并为一个字符串
    input_id = torch.tensor([tokenizer.encode(input_for_prm)])  # 对合并后的字符串进行编码
    # 通过调用tokenizer.encode方法完成的，该方法将字符串转换为一个数字序列，其中每个数字代表词汇表中的一个单词或符号
    # 如果你直接将这个列表传递给torch.tensor()，你会得到一个一维的张量。但是，在许多深度学习模型（特别是处理自然语言的模型）中，输入通常需要是二维的，即使这个二维张量的其中一个维度只有一个元素。所以，[tokenizer.encode(input_for_prm)]是将整数列表包装在另一个列表中，这样torch.tensor()会返回一个二维张量，其中第一个维度的大小为1，第二个维度的大小为编码的长度。这样做的原因是，许多模型的输入需要是批量的数据，即使只有一个输入序列，也需要将其视为一个批量大小为1的批量。这样，无论输入一个还是多个序列，模型的代码都可以保持一致。
    # 由于transformers库中的模型接受的输入是一个整数张量，所以我们需要将字符串转换为整数张量。这里使用torch.tensor方法将字符串转换为整数张量。比如torch.tensor([1, 2, 3, 4, 5])返回的是 tensor([1, 2, 3, 4, 5])

    with torch.no_grad():  # 不计算梯度：这在只想进行前向传播，而不想计算任何梯度时非常有用，例如在模型推理（预测）阶段。（在训练神经网络时，我们需要计算梯度以更新模型的参数。然而，在推理阶段，我们只是使用模型进行预测，不需要更新模型的参数，因此也就不需要计算梯度。禁止梯度计算可以减少内存使用量，并且可能会加速计算，因为不需要进行反向传播。）
        logits = model(input_id).logits[:,:,candidate_tokens]  # 通过模型获取对数几率
        # 首先，model(input_id)是将输入数据input_id传递给模型进行预测。这将返回一个包含多个属性的输出对象，其中一个属性是logits。logits是模型在最后一层的输出，它们是未归一化的预测值，也就是说，它们的值可以是任何实数，不仅仅是0到1之间。
        # 然后，我们使用[:, :, candidate_tokens]来获取我们感兴趣的标记的对数几率。这里的[:, :, candidate_tokens]是一个切片操作，它的作用是从模型的输出中选择我们感兴趣的标记。在这里，我们只关心正面和负面标记的对数几率，所以我们使用candidate_tokens来选择这两个标记。这样，logits的形状将变为(batch_size, sequence_length, num_candidate_tokens)，其中num_candidate_tokens是我们感兴趣的标记的数量。

        scores = logits.softmax(dim=-1)[:,:,0]  # 计算softmax得分
        # 首先，我们使用softmax函数对对数几率进行归一化，以得到标记的概率分布。softmax函数的作用是将实数转换为概率分布，使得所有元素的和等于1。这样，我们就可以得到每个标记的概率，而不仅仅是对数几率。dim=-1 表示对最后一个维度进行 softmax 操作，这里的最后一个维度是 num_candidate_tokens，也就是我们感兴趣的标记的数量。
        # 然后，我们使用[:, :, 0]来获取我们感兴趣的标记的概率。这里的[:, :, 0]是一个切片操作，它的作用是从概率分布中选择我们感兴趣的标记。在这里，我们只关心正面标记的概率，所以我们使用0来选择这个标记。这样，scores的形状将变为(batch_size, sequence_length)，其中sequence_length是输入序列的长度。
        
        step_scores = scores[input_id == step_tag_id]  # 获取步骤标记对应的得分
        # 首先，我们使用input_id == step_tag_id来获取步骤标记对应的位置。这里的input_id == step_tag_id是一个逻辑运算，它的作用是将input_id中等于step_tag_id的位置设置为True，其余位置设置为False。这样，我们就可以得到步骤标记对应的位置。
        # 然后，我们使用scores[input_id == step_tag_id]来获取步骤标记对应的得分。这里的scores[input_id == step_tag_id]是一个切片操作，它的作用是从得分中选择步骤标记对应的得分。这样，step_scores的形状将变为(batch_size,)，其中batch_size是输入序列的长度。
        
        print(step_scores)  # 打印得分

# 输出：      
# tensor([0.9955, 0.9958, 0.9983, 0.9957])
# tensor([0.9955, 0.9958, 0.9983, 0.0240]) 
