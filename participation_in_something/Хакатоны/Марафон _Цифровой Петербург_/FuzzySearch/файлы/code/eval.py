import json
import onnxruntime
import numpy as np
from tokenizers import Tokenizer



def get_word_preds(inputs_ids):

    tokens = [tokenizer.id_to_token(id) for id in inputs_ids]
    #print(tokens)

    words = []
    classes = []
    current_word = ''
    punctuation = ['.', '!', '?', ';', ':']

    for idx, token in enumerate(tokens):
        if token in ['[CLS]', '[PAD]']:
            continue
        if token == '[SEP]':
            break



        # если токен — знак препинания
        if token in punctuation or token.startswith('##') and token[2:] in punctuation:
            continue
            # if current_word:
            #     if predictions[0][idx-1].item() != 0:
            #         words.append(current_word)
            #         classes.append(predictions[0][idx-1].item())
            #         current_word = ''
            # if len(words) > 0:
            #     words[-1] += token.replace('##', '')  # приклеиваем знак препинания
            # continue



        if token.startswith('##'):
            current_word += token[2:]
        else:
            if current_word:
                if predictions[0][idx-1].item() != 0:
                    words.append(current_word)
                    classes.append(predictions[0][idx-1].item())
            current_word = token

    if current_word:
        if predictions[0][idx-1].item() != 0:
            words.append(current_word)
            classes.append(predictions[0][idx-1].item())

    return words, classes

def merge_entities(words, classes):
    merged_words = []
    merged_classes = []
    i = 0
    while i < len(classes):
        cls = classes[i]
        if cls % 2 == 1:  # Начало сущности (B-)
            entity_words = [words[i]]
            base_cls = cls
            i += 1
            # Добавляем все последующие I- метки той же сущности
            while i < len(classes) and classes[i] == base_cls + 1:
                entity_words.append(words[i])
                i += 1
            merged_words.append(' '.join(entity_words))
            merged_classes.append(base_cls)
        else:
            # Пропускаем одиночные I- или O
            i += 1
    return merged_words, merged_classes





# ___Загрузка модели_________________________________________________________

# current_dir = os.path.dirname(os.path.abspath(__file__))
# model_path = os.path.join(current_dir, "classificationModel")


session = onnxruntime.InferenceSession("./onnxModel/midMybert.onnx")
tokenizer = Tokenizer.from_file("./tokenizer/tokenizer.json")





# _____токенизация______________________________________________________


# Текст
text = "ул. Маяковская, 48, литера А"

# Токенизация
encoded = tokenizer.encode(text)

# Преобразование в нужный формат
input_ids = encoded.ids
attention_mask = encoded.attention_mask

# Приведение к фиксированной длине (например, 128)
max_length = 128
pad_id = tokenizer.token_to_id("[PAD]")  # Убедись, что у тебя есть [PAD] в словаре

input_ids += [pad_id] * (max_length - len(input_ids))
attention_mask += [0] * (max_length - len(attention_mask))


# Преобразуем в numpy массивы
input_ids = np.array([input_ids], dtype=np.int64)
attention_mask = np.array([attention_mask], dtype=np.int64)








#____Инференс_________________________________________________________

# Получение выходов
outputs = session.run(
    None,
    {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }
)

# Результат
logits = outputs[0]

# Берём argmax по логитам
predictions = np.argmax(logits, axis=-1)
#print(predictions)


words, classes = get_word_preds(input_ids[0])
#print(words)
#print(classes)


merged_words, merged_classes = merge_entities(words, classes)
#print(merged_words)
#print(merged_classes)


import json
data = [{'word': w, 'label': l} for w, l in zip(merged_words, merged_classes)]
json_string = json.dumps(data, ensure_ascii=False, indent=2)
print(json_string)