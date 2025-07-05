import json

def convert_sharegpt_format(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    new_data = []
    for dialogue in data:
        convs = dialogue.get("conversations", [])
        new_conv = []
        for msg in convs:
            role = "user" if msg["from"] == "human" else "assistant"
            new_conv.append({
                "role": role,
                "content": msg["value"]
            })
        new_data.append(new_conv)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)

# 使用方式示例
convert_sharegpt_format("ShareGPT_Vicuna_unfiltered/ShareGPT_V4.3_unfiltered_cleaned_split.json", "ShareGPT.json")